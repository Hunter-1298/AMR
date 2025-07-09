import torch
from scipy import signal
import h5py
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
import os
import pickle
from collections import defaultdict
from tqdm import tqdm


class RFMLDataset(Dataset):
    def __init__(
        self,
        dataPath="/home/hshayde/Projects/MIT/AMR/Dataset/RML2016.10a_dict.pkl",
        data=2018,
        iq=False,
        sync=False
    ):
        # Data in the shape of dict[('Mod_type','snr')] = [1000,2,128]
        if data == 2018:
            data = self._load_2018_data(sync)
        else:
            data = self._load_data(dataPath)

        # Convert data to tensors and split
        self.samples = []
        self.sync_samples = []
        self.labels = []
        self.snr = []
        self.encoded_hash = {}

        # Remove the pdb line for production
        # import pdb; pdb.set_trace()

        for (mod_type, snr_val), signal_data in data.items():
            if sync:
                # When sync=True, signal_data is a list of tuples: [(synced, original), ...]
                sync_signals_list = []
                original_signals_list = []

                for sync_sig, orig_sig in signal_data:
                    sync_signals_list.append(sync_sig)
                    original_signals_list.append(orig_sig)

                # Convert to tensors
                sync_signals = torch.from_numpy(np.array(sync_signals_list)).float()
                original_signals = torch.from_numpy(np.array(original_signals_list)).float()

            else:
                # When sync=False, signal_data is a list of arrays: [signal, signal, ...]
                original_signals = torch.from_numpy(np.array(signal_data)).float()
                sync_signals = original_signals  # Same as original when no sync data

            mod_label = mod_type

            # Normalize all signals at once
            if iq:
                processed_original = self._normalize_data(original_signals)
                processed_sync = self._normalize_data(sync_signals)
            else:
                # Convert to amplitude/phase for all signals at once
                processed_original = self._process_signals(original_signals)
                processed_sync = self._process_signals(sync_signals)

            # Extend lists with all samples at once
            self.samples.extend(list(processed_original))
            self.sync_samples.extend(list(processed_sync))

            labels = [self._encode_labels(mod_label)] * original_signals.shape[0]
            self.labels.extend(labels)
            self.snr.extend([snr_val] * original_signals.shape[0])

        # create decoded hash to convert back
        self._decode_labels(self)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sync_samples[idx], self.samples[idx], self.labels[idx], self.snr[idx]

    def _load_data(self, dataPath):
        with open(dataPath, "rb") as f:
            data = pickle.load(f, encoding="latin")
        return data


    def _sync(self, x, mod_type, plot=False):
        # Convert (2, 1024) to complex: x[0] = real, x[1] = imag
        # Convert to complex
        x_complex = x[0] + 1j * x[1]

        # Modulation-specific parameters
        mod_params = {
            'QPSK':  {'sps': 8, 'mod_order': 4, 'costas_bw': 0.01,  'costas_damp': 0.707},
            '8PSK':  {'sps': 8, 'mod_order': 8, 'costas_bw': 0.005, 'costas_damp': 0.707},
            '16PSK': {'sps': 8, 'mod_order': 16, 'costas_bw': 0.0025, 'costas_damp': 0.707},
        }
        assert mod_type in mod_params, f"Unsupported modulation: {mod_type}"
        params = mod_params[mod_type]
        sps, mod_order = params['sps'], params['mod_order']

        def mm_timing_sync(samples, sps, gain=0.3, modulation_order=4):
            mu = 0.0
            N = len(samples)
            out = np.zeros(N + 10, dtype=np.complex64)
            out_rail = np.zeros_like(out)
            i_in = 0
            i_out = 2
            sps_up = sps * 16
            constellation = np.exp(1j * 2 * np.pi * np.arange(modulation_order) / modulation_order)

            while (i_out < N) and (i_in + sps_up < N):
                idx = int(i_in + mu)
                frac = mu - int(mu)
                s0 = samples[idx]
                s1 = samples[idx + 1]
                sample = s0 * (1 - frac) + s1 * frac

                out[i_out] = sample
                out_rail[i_out] = constellation[np.argmin(np.abs(sample - constellation))]

                x_err = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
                y_err = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
                mm_val = np.real(y_err - x_err)

                mu += sps_up + gain * mm_val
                i_in += int(mu)
                mu = mu - int(mu)
                i_out += 1

            return out[2:i_out]

        def costas_loop(signal, loop_bandwidth, damping_factor, modulation_order):
            N = len(signal)
            phase_est = 0.0
            freq_est = 0.0
            output = np.zeros(N, dtype=np.complex64)
            theta = loop_bandwidth / (damping_factor + 0.25 / damping_factor)
            d = 1 + 2 * damping_factor * theta + theta**2
            alpha = (4 * damping_factor * theta) / d
            beta = (4 * theta**2) / d
            constellation = np.exp(1j * 2 * np.pi * np.arange(modulation_order) / modulation_order)

            for n in range(N):
                corrected = signal[n] * np.exp(-1j * phase_est)
                output[n] = corrected
                nearest = constellation[np.argmin(np.abs(corrected - constellation))]
                error = np.angle(corrected * np.conj(nearest))
                freq_est += beta * error
                phase_est += freq_est + alpha * error

            return output


        # Step 1: interpolate for M&M
        x_interp = signal.resample_poly(x_complex, up=16, down=1)

        # Step 2: M&M timing sync
        x_mm = mm_timing_sync(x_interp, sps=8, modulation_order=mod_order)

        # Step 3: Costas loop carrier sync
        x_costa = costas_loop(
            x_mm,
            loop_bandwidth=params['costas_bw'],
            damping_factor=params['costas_damp'],
            modulation_order=mod_order
        )

        # Step 4: resample to original 1024
        x_sync = signal.resample(x_costa, 1024)

        # Optional: plot synced constellation
        if plot:
            plt.figure(figsize=(15, 5))

            # Plot original unsynced signal
            plt.subplot(1, 3, 1)
            plt.scatter(x_complex.real, x_complex.imag, s=2, alpha=0.6)
            plt.title(f'{mod_type} - Original Unsynced')
            plt.grid(True)
            plt.axis('equal')

            # Plot upsampled synced signal (x_mm or x_costas - pick Costas output here)
            plt.subplot(1, 3, 2)
            plt.scatter(x_costa.real, x_costa.imag, s=2, alpha=0.6)
            plt.title(f'{mod_type} - Upsampled Synced (After Costas)')
            plt.grid(True)
            plt.axis('equal')

            # Plot final synced and resampled back to 1024
            plt.subplot(1, 3, 3)
            plt.scatter(x_sync.real, x_sync.imag, s=2, alpha=0.6)
            plt.title(f'{mod_type} - Synced & Resampled (1024 samples)')
            plt.grid(True)
            plt.axis('equal')

            plt.tight_layout()
            plt.show()

        # Step 5: convert to (2, 1024)
        return np.stack([x_sync.real, x_sync.imag], axis=0)


    def _load_2018_data(self, sync):
        classes = [
            "OOK",
            "4ASK",
            "8ASK",
            "BPSK",
            "QPSK",
            "8PSK",
            "16PSK",
            "32PSK",
            "16APSK",
            "32APSK",
            "64APSK",
            "128APSK",
            "16QAM",
            "32QAM",
            "64QAM",
            "128QAM",
            "256QAM",
            "AM-SSB-WC",
            "AM-SSB-SC",
            "AM-DSB-WC",
            "AM-DSB-SC",
            "FM",
            "GMSK",
            "OQPSK",
        ]
        data_path = "/home/hshayde/Projects/MIT/AMR/Dataset/2018.01/2018_RFML.hdf5"
        data_dict = {}
        # choosen_classes = [
        #     "QPSK",
        #     "16QAM",
        #     "64QAM",
        #     "OOK",
        #     "8PSK",
        #     "16PSK",
        #     "AM-SSB-SC",
        #     "AM-DSB-WC",
        #     "FM",
        #     "BPSK",
        #     "GMSK",
        # ]
        choosen_classes = ["QPSK", "8PSK", "16PSK"]
        min_snr_level = -5
        if sync: # load synchronized data, should be a dict of synchonized data
            sync_data_path = '/home/hshayde/Projects/MIT/AMR/Dataset/sync_data.pkl'
            if not os.path.exists(sync_data_path):
                print(f"Synchronized data file not found at {sync_data_path}, creating the data and syncing manually")
                # If we dont have sync we can load from memory
                with h5py.File(data_path, "r") as f:
                    X = f["X"][:]  # [num_samples, 2, signal_length]
                    Y = f["Y"][:]  # [num_samples]
                    Z = f["Z"][:]  # [num_samples]
                    # Decode byte labels to string if necessary
                    if isinstance(Y[0], bytes):
                        Y = [y.decode("utf-8") for y in Y]
                    for x, y, z in tqdm(zip(X, Y, Z)):
                        if classes[np.argmax(y)] in choosen_classes and int(z) >= min_snr_level:
                            key = (classes[np.argmax(y)], int(z))  # (mod_type, snr) key
                            if key not in data_dict:
                                data_dict[key] = []
                            data_dict[key].append((self._sync(x.T, key[0]), x.T))  # transpose so channels x features
                with open(sync_data_path, 'wb') as f:
                    print('Saving synced data into new file')
                    pickle.dump(data_dict, f)
            else: # load the sync data
                print(f' Loading synced Data')
                with open(sync_data_path, 'rb') as f:
                    data_dict = pickle.load(f)

        else: #  Load unsync'd data
            # If we dont have sync we can load from memory
            with h5py.File(data_path, "r") as f:
                X = f["X"][:]  # [num_samples, 2, signal_length]
                Y = f["Y"][:]  # [num_samples]
                Z = f["Z"][:]  # [num_samples]
                # Decode byte labels to string if necessary
                if isinstance(Y[0], bytes):
                    Y = [y.decode("utf-8") for y in Y]
                for x, y, z in tqdm(zip(X, Y, Z)):
                    if classes[np.argmax(y)] in choosen_classes and int(z) >= min_snr_level:
                        key = (classes[np.argmax(y)], int(z))  # (mod_type, snr) key
                        if key not in data_dict:
                            data_dict[key] = []
                        data_dict[key].append(x.T)  # transpose so channels x features
                print(f"Total keys created: {len(data_dict)}")
                total_samples = sum(len(v) for v in data_dict.values())
                print(f"Total signals stored: {total_samples}")
        return data_dict

    def _encode_labels(self, label):
        # takes samples and return one hot encoding
        if label not in self.encoded_hash.keys():
            self.encoded_hash[label] = len(self.encoded_hash)
        return self.encoded_hash[label]

    def _decode_labels(self, label):
        # invert the encoding
        self.decoded_hash = {value: key for key, value in self.encoded_hash.items()}

    # def _normalize_data(self, data):
    #     # data shape: [batch_size, 2, 128] or [2, 128]
    #     # if len(data.shape) == 2:
    #     #     power = torch.sqrt(torch.sum(data**2, dim=1, keepdim=True))
    #     # else:  # batch mode
    #     #     power = torch.sqrt(torch.sum(data**2, dim=2, keepdim=True))
    #     # return data / (power + 1e-8)  # Add small epsilon to avoid division by zero
    #     dims = (0, 2) if data.dim() == 3 else (1,)
    #     mean = data.mean(dim=dims, keepdim=True)
    #     std  = data.std(dim=dims, keepdim=True)
    #     return (data - mean) / (std + 1e-8)
    def _normalize_data(self, data):
        # data shape: [batch_size, 2, 128] or [2, 128]

        if data.dim() == 3:  # batch mode [batch_size, 2, 128]
            # Normalize each sample independently
            normalized = torch.zeros_like(data)
            for i in range(data.shape[0]):
                sample = data[i]  # [2, 128]
                sample_min = sample.min()
                sample_max = sample.max()
                sample_range = sample_max - sample_min
                normalized[i] = 2 * (sample - sample_min) / (sample_range + 1e-8) - 1
            return normalized

        else:  # single sample [2, 128]
            data_min = data.min()
            data_max = data.max()
            data_range = data_max - data_min
            return 2 * (data - data_min) / (data_range + 1e-8) - 1

    def _process_signals(self, signals):
        i_data = signals[:, 0, :]
        q_data = signals[:, 1, :]

        # First compute amplitude and phase
        amplitude = torch.sqrt(i_data**2 + q_data**2)
        phase = torch.atan2(q_data, i_data)

        # Then normalize amplitude separately to maintain its variation
        amplitude = amplitude / (torch.max(amplitude, dim=1, keepdim=True)[0] + 1e-8)
        # Phase is already normalized by nature (-π to π)

        return torch.stack([amplitude, phase], dim=1)

    def get_decoded_labels(self):
        # Return the mapping of encoded indices to original modulation labels
        return self.decoded_hash


def get_dataloaders(config):
    # Create full dataset
    full_dataset = RFMLDataset(data=config.data, iq=config.iq, sync=config.sync)

    # Get parameters from config
    batch_size = config.batch_size
    num_workers = config.num_workers
    train_val_split = config.train_val_split
    random_seed = config.random_seed

    # Calculate split sizes
    train_size = int(train_val_split[0] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Perform random split
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed),  # for reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the training data
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
    )

    # Get the mapping of indices to modulation labels
    mod_names = full_dataset.get_decoded_labels()

    return train_loader, val_loader, mod_names


class TokenizedRFMLDataset(Dataset):
    def __init__(self, base_dataset, vqvae):
        self.base_dataset = base_dataset
        self.vqvae = vqvae
        self.vqvae.eval()

        self.tokenized_data = []
        self.labels = []
        self.snrs = []

        # Create a temporary dataloader for batch processing
        loader = DataLoader(
            base_dataset,
            batch_size=64,  # Larger batch size for faster processing
            shuffle=False,
            num_workers=4,
        )

        # Preload all data
        with torch.no_grad():
            for samples, labels, snrs in tqdm(loader, desc="Tokenizing dataset"):
                # returns [batch_size, num_channels =2, num_tokens, embed_dim]
                quantized = vqvae.encode(samples)
                self.tokenized_data.extend(quantized.cpu())
                self.labels.extend(labels.cpu())
                self.snrs.extend(snrs.cpu())

            # Convert lists to tensors for faster access
            self.tokenized_data = torch.stack(self.tokenized_data)
            self.labels = torch.tensor(self.labels)
            self.snrs = torch.tensor(self.snrs)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.tokenized_data[idx], self.labels[idx], self.snrs[idx]


def get_tokenized_dataloaders(cfg, vqvae, train_loader, val_loader):
    """Creates new dataloaders with tokenized datasets"""

    # Create tokenized datasets
    print("Creating training dataset...")
    tokenized_train_dataset = TokenizedRFMLDataset(
        train_loader.dataset,
        vqvae,
    )

    print("Creating validation dataset...")
    tokenized_val_dataset = TokenizedRFMLDataset(
        val_loader.dataset,
        vqvae,
    )

    # Get parameters from cfg
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    # Create new dataloaders
    tokenized_train_loader = DataLoader(
        tokenized_train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
    )

    tokenized_val_loader = DataLoader(
        tokenized_val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
    )

    return tokenized_train_loader, tokenized_val_loader


class MoCoRFMLDataset(Dataset):
    def __init__(self, dataset, high_snr_threshold=20):
        self.dataset = dataset
        self.n = len(dataset)

        # Build mapping: modulation -> list of high-SNR indices
        self.high_snr_indices = defaultdict(list)
        for idx in range(self.n):
            _, _, mod, snr = dataset[idx]
            if snr >= high_snr_threshold:
                self.high_snr_indices[mod].append(idx)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x1_sync, x1, mod, snr1 = self.dataset[idx]
        # Get high-SNR positive sample with same modulation
        candidates = self.high_snr_indices.get(mod, [])
        pos_idx = random.choice(candidates)
        x2_sync, x2, _, snr2 = self.dataset[pos_idx]
        return (x1_sync, x2_sync), mod, (snr1, snr2)


def get_moco_dataloaders(train_loader, val_loader, config):
    """
    Creates MoCo dataloaders from existing train/val splits

    Args:
        train_loader: Original training dataloader
        val_loader: Original validation dataloader
        config: Configuration object
    """
    # Create MoCo datasets from existing splits
    moco_train_dataset = MoCoRFMLDataset(train_loader.dataset)
    moco_val_dataset = MoCoRFMLDataset(val_loader.dataset)

    # Create new dataloaders with MoCo datasets
    moco_train_loader = DataLoader(
        moco_train_dataset,
        batch_size=config.contrastive_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    moco_val_loader = DataLoader(
        moco_val_dataset,
        batch_size=config.contrastive_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Get label names from original dataset
    label_names = train_loader.dataset.dataset.get_decoded_labels()

    return moco_train_loader, moco_val_loader, label_names
