import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
import os
import pickle
from collections import defaultdict
from tqdm import tqdm


class RFMLDataset(Dataset):
    def __init__(self, dataPath = '/workspace/hhayden/AMR/Dataset/RML2016.10a_dict.pkl', iq=False):
        # Data in the shape of dict[('Mod_type','snr')] = [1000,2,128]
        data = self._load_data(dataPath)

        # Get first key to determine total samples
        first_key = list(data.keys())[0]
        total_samples = data[first_key].shape[0]

        # Convert data to tensors and split
        self.samples = []
        self.labels = []
        self.snr = []
        self.encoded_hash = {}

        for (mod_type, snr), signals in data.items():
            signals = torch.from_numpy(signals).float()  # signals shape: [1000, 2, 128]
            mod_label = mod_type
            # Normalize all signals at once
            if iq:
                processed_signals = self._normalize_data(signals)  # Now shape: [1000, 2, 128]
            else:
                # Convert to amplitude/phase for all signals at once
                processed_signals = self._process_signals(signals)

            # Extend lists with all samples at once
            self.samples.extend(list(processed_signals))
            labels = [self._encode_labels(mod_label)] * signals.shape[0]
            self.labels.extend(labels)
            self.snr.extend([snr] * signals.shape[0])

            # create decoded hash to convert back
            self._decode_labels(self)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx],self.snr[idx]


    def _load_data(self, dataPath):
        with open(dataPath, 'rb') as f:
            data = pickle.load(f, encoding="latin")
        return data

    def _encode_labels(self,label):
        # takes samples and return one hot encoding
        if label not in self.encoded_hash.keys():
            self.encoded_hash[label] = len(self.encoded_hash)
        return self.encoded_hash[label]

    def _decode_labels(self,label):
        # invert the encoding
        self.decoded_hash = {value:key for key,value in self.encoded_hash.items()}

    def _normalize_data(self, data):
        # data shape: [batch_size, 2, 128] or [2, 128]
        # if len(data.shape) == 2:
        #     power = torch.sqrt(torch.sum(data**2, dim=1, keepdim=True))
        # else:  # batch mode
        #     power = torch.sqrt(torch.sum(data**2, dim=2, keepdim=True))
        # return data / (power + 1e-8)  # Add small epsilon to avoid division by zero
        dims = (0, 2) if data.dim() == 3 else (1,)
        mean = data.mean(dim=dims, keepdim=True)
        std  = data.std(dim=dims, keepdim=True)
        return (data - mean) / (std + 1e-8)

    def _process_signals(self, signals):
        i_data = signals[:,0,:]
        q_data = signals[:,1,:]

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
    full_dataset = RFMLDataset(iq = config.iq)

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
        generator=torch.Generator().manual_seed(random_seed)  # for reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the training data
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers
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
            num_workers=4
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
        num_workers=num_workers
    )

    tokenized_val_loader = DataLoader(
        tokenized_val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers
    )

    return tokenized_train_loader, tokenized_val_loader
