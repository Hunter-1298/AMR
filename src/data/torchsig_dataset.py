import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator, default_collate_fn, batch_as_signal_list
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.transforms.transforms import Spectrogram

import torch
import numpy as np

def custom_collate(batch):
    """
    Collate function for IQ signal data with batch-level normalization.

    Each batch element:
        (data, [label, snr])

    data: complex-valued numpy array (I + jQ)
    label: class label
    snr: signal-to-noise ratio

    Returns:
        data_tensor: [batch_size, 2, num_samples] float32 tensor
        (label_tensor, snr_tensor): tuple of tensors
    """
    data_list, label_list, snr_list = [], [], []

    # First, split I/Q and stack for each sample
    for signal, (label, snr) in batch:
        I = np.real(signal)
        Q = np.imag(signal)
        iq = np.stack([I, Q], axis=0)  # [2, num_samples]
        data_list.append(iq)
        label_list.append(label)
        snr_list.append(snr)

    # --- Convert lists to arrays for batch-level normalization ---
    data_array = np.stack(data_list, axis=0)  # [batch_size, 2, num_samples]

    # --- Batch-level normalization ---
    max_val = np.max(np.abs(data_array))
    data_array = data_array / (max_val + 1e-8)

    # --- Convert to tensors ---
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    snr_tensor = torch.tensor(snr_list, dtype=torch.float32)

    return data_tensor, (label_tensor, snr_tensor)


def get_torchsig_dataloader(
    train_root: str = "/weka/hhayden/AMR/datasets/torchsig_train",
    test_root: str = "/weka/hhayden/AMR/datasets/torchsig_test",
    class_list = ['bpsk', 'qpsk', '8psk', '16psk', '32psk', '64psk'],
    # class_list = [ 'qpsk', 'bpsk', '8psk', '16psk'], 
    # class_list = [ 'qpsk', 'bpsk'],
    # class_list = [ 'qpsk', 'bpsk', '8psk' ],
    # class_list = [ 'qpsk', 'bpsk', '8psk', '16psk'],
    train_size: int = 100000,
    test_size: int = 20000,
    batch_size: int = 64,
    fft_size: int = 64,
    num_iq_samples_dataset: int = 4096,
    signal_bandwidth_min: int = 1e6,     
    signal_bandwidth_max: int = 1e6,
    signal_center_freq: int = 0,
    num_signals_min: int = 2,
    num_signals_max: int = 2,
    snr_db_min: float = 0.0,
    snr_db_max: float = 30.0,
    sample_rate=10e6,
    seed: int = 123456789
):
    def make_or_load_dataset():
        # Create dataset if not found
        metadata = DatasetMetadata(
            num_iq_samples_dataset=num_iq_samples_dataset,
            fft_size=fft_size,
            num_signals_min=num_signals_min,
            num_signals_max=num_signals_max,
            snr_db_min = snr_db_min,
            snr_db_max = snr_db_max,
            impairment_level=0,
            class_list = class_list,
            signal_duration_min=float(num_iq_samples_dataset/sample_rate),
            signal_duration_max=float(num_iq_samples_dataset/sample_rate),
            signal_bandwidth_min = signal_bandwidth_min,     # 1 MHz,
            signal_bandwidth_max = signal_bandwidth_min,
            signal_center_freq_min = signal_center_freq,
            signal_center_freq_max = signal_center_freq,
        )

        train_dataset = TorchSigIterableDataset(
            dataset_metadata=metadata,
            # transforms = []
        )
        test_dataset = TorchSigIterableDataset(
            dataset_metadata=metadata,
            # transforms = []
        )

        train_dataloader = WorkerSeedingDataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn = lambda x: x
        )
        # train_dataloader.seed(seed)
        test_dataloader = WorkerSeedingDataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn = lambda x: x
        )
        # test_dataloader.seed(seed)

        train_dataset_creator = DatasetCreator(
            dataloader=train_dataloader,
            root=train_root,
            overwrite=True,
            dataset_length = train_size,
            multithreading=False
        )
        test_dataset_creator = DatasetCreator(
            dataloader=test_dataloader,
            root=test_root,
            overwrite=True,
            dataset_length = test_size,
            multithreading=False
        )
        if not os.path.exists(train_root) or not os.path.exists(test_root):
            train_dataset_creator.create()
            test_dataset_creator.create()

        train_dataset = StaticTorchSigDataset(
            root = train_root,
            target_labels=["class_index", "snr_db"]
        )
        test_dataset = StaticTorchSigDataset(
            root = test_root,
            target_labels=["class_index", "snr_db"]
        )
        train_dataloader = WorkerSeedingDataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=batch_size//2,
            collate_fn = custom_collate
        )

        test_dataloader = WorkerSeedingDataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=batch_size//2,
            collate_fn = custom_collate
        )

        return train_dataloader, test_dataloader

        # if not os.path.exists(root) or not os.listdir(root):
        #     print(f"Creating dataset at: {root}")
        #     dataset_creator.create()
        # else:
            # print(f"Found existing dataset at: {root}")
            # Load from disk
            # static_dataset = StaticTorchSigDataset(
            #     root=root,
            # )

    # Create or load both train and test datasets
    train_dataset, test_dataset = make_or_load_dataset()
    label_names = {}
    i = 0
    for mod in class_list:
        label_names[i] = mod
        i +=1


    return train_dataset, test_dataset, label_names

       