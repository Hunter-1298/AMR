import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
def _normalize_data(data):
    if data.dim() == 2:
        power = torch.mean(data ** 2)
        return data / torch.sqrt(power + 1e-8)
    elif data.dim() == 3:
        power = torch.mean(data ** 2, dim=[1, 2], keepdim=True)  # shape: [batch, 1, 1]
        return data / torch.sqrt(power + 1e-8)
    # # data shape: [batch_size, 2, 128] or [2, 128]
    # if data.dim() == 3:  # batch mode [batch_size, 2, 128]
    #     # Normalize each sample independently
    #     normalized = torch.zeros_like(data)
    #     for i in range(data.shape[0]):
    #         sample = data[i]  # [2, 128]
    #         sample_min = sample.min()
    #         sample_max = sample.max()
    #         sample_range = sample_max - sample_min
    #         normalized[i] = 2 * (sample - sample_min) / (sample_range + 1e-8) - 1
    # return normalized

def _load_data(dataPath):
    with open(dataPath, 'rb') as f:
        data = pickle.load(f, encoding="latin")
    return data
# Load in the data
dataPath = '/home/hshayde/Projects/MIT/AMR/Dataset/RML2016.10a_dict.pkl'
# Data in the shape of dict[('Mod_type','snr')] = [1000,2,128]
data = _load_data(dataPath)
print('Data Loaded Successfully')
# Get dataset info
first_key = list(data.keys())[0]
total_samples_per_key = data[first_key].shape[0]
print(f"Samples per (modulation, SNR): {total_samples_per_key}")
print(f"Total keys (mod_type, SNR combinations): {len(data.keys())}")
# Process each modulation type and SNR combination
# Initialize lists
samples = {}
mod_types = set()
print("Processing dataset...")
for (mod_type, snr_val), signals in tqdm(data.items()):
    # Convert to tensor
    mod_types.add(mod_type)
    signals = torch.from_numpy(signals).float()  # signals shape: [1000, 2, 128]
    processed_signals = _normalize_data(signals)
    samples[(mod_type, snr_val)] = processed_signals
# Choose normalization method
# Show all of our modulation types
print(mod_types)
# we want to look QPSK
QPSK_18DB = samples[('QPSK', 18)]
QAM16_18DB = samples[('QAM16', 18)]
QAM64_18DB = samples[('QAM64', 18)]
# Lets see how much data we have
print(f' Number of Samples: {len(QPSK_18DB)}, of shape: {QPSK_18DB.shape[1:]}') # in the for of [1000,2,128]
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 12))

# Data for the three modulation types
mod_data = [QPSK_18DB, QAM16_18DB, QAM64_18DB]
mod_names = ['QPSK', 'QAM16', 'QAM64']
plt.figure(figsize=(10,10))
for mod_idx, (data, mod_name) in enumerate(zip(mod_data, mod_names)):
    for sample_idx in range(3):
        plt.subplot(3, 3, mod_idx * 3 + sample_idx + 1)
        # Extract I and Q channels for each sample
        sample = data[sample_idx][0] +1j* data[sample_idx][1]
        # Create constellation diagram
        plt.scatter(sample.real, sample.imag, alpha=0.6, s=20)
        plt.title(f'{mod_name} Sample {sample_idx+1} at 18 dB SNR')
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

plt.tight_layout()
plt.show()


def mm_timing_sync(samples, sps=64, gain=0.3):
    """
    Mueller and Müller timing recovery for QPSK signals.

    Args:
        iq_data: ndarray of shape (N, 2), with [I, Q] columns
        sps: samples per symbol (after interpolation). Typical: 64 if 16x upsampled and true SPS is 4.
        gain: loop gain for timing adjustment

    Returns:
        synced: complex NumPy array of aligned samples
    """
    # Convert I/Q to complex baseband

    mu = 0  # Initial phase estimate
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)

    i_in = 0  # Index in input samples
    i_out = 2  # Output index (start from 2 due to need for delay)

    while i_out < len(samples) and i_in + 16 < len(samples):
        out[i_out] = samples[i_in]
        out_rail[i_out] = (int(np.real(out[i_out]) > 0) +
                        1j * int(np.imag(out[i_out]) > 0))

        x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
        y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
        mm_val = np.real(y - x)

        mu += sps + gain * mm_val
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu)
        i_out += 1

    return out[2:i_out]  # Discard initial placeholder samples
def costas_loop(signal, loop_bandwidth=0.01, damping_factor=0.707, modulation_order=4):
    """
    Costas Loop for carrier phase recovery in QPSK signals.
    Args:
        signal: Complex baseband input (e.g., after timing sync)
        loop_bandwidth: Determines loop responsiveness
        damping_factor: Controls damping in the loop filter
        modulation_order: 4 for QPSK
    Returns:
        corrected_signal: Phase-corrected complex samples
    """
    N = len(signal)
    phase_est = 0.0
    freq_est = 0.0
    output = np.zeros(N, dtype=np.complex64)

    # Loop filter coefficients (2nd order PLL)
    Kp = 1.0  # Phase detector gain
    K0 = 1.0  # VCO gain
    theta = loop_bandwidth / (damping_factor + 0.25 / damping_factor)
    d = 1 + 2 * damping_factor * theta + theta**2
    alpha = (4 * damping_factor * theta) / d
    beta = (4 * theta**2) / d

    for n in range(N):
        # Rotate input by negative estimated phase
        corrected = signal[n] * np.exp(-1j * phase_est)
        output[n] = corrected

        # Decision-directed error
        if modulation_order == 4:
            # QPSK phase error from symbol decision
            error = np.sign(corrected.real) * corrected.imag - np.sign(corrected.imag) * corrected.real
        else:
            raise NotImplementedError("Only QPSK (M=4) is implemented.")

        # Loop filter
        freq_est += beta * error
        phase_est += freq_est + alpha * error

    return output
# Now we can see that we are not time synces, lets play around with pySDR to lock in these time syncrhinizations first and then go from there
QPSK_COMPLEX = QPSK_18DB[0][0] + 1j *  QPSK_18DB[0][1]
from scipy import signal
# Oversample (interpolate) by 16x
samples_interpolated = signal.resample_poly(QPSK_COMPLEX, up=16, down=1)
synced_qpsk = mm_timing_sync(samples_interpolated, sps=8)
synced_qpsk = costas_loop(synced_qpsk)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Plot original signal (not interpolated/synced)
ax1.plot(QPSK_COMPLEX.real, QPSK_COMPLEX.imag, '.', alpha=0.5)
ax1.set_title("Original QPSK Signal")
ax1.set_xlabel('In-Phase (I)')
ax1.set_ylabel('Quadrature (Q)')
ax1.grid(True)
ax1.axis('equal')
# Plot interpolated and synced signal
ax2.plot(synced_qpsk.real, synced_qpsk.imag, '.', alpha=0.5)
ax2.set_title("QPSK After Timing Recovery with Interpolation")
ax2.set_xlabel('In-Phase (I)')
ax2.set_ylabel('Quadrature (Q)')
ax2.grid(True)
ax2.axis('equal')
plt.tight_layout()
plt.show()
