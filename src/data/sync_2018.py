import torch
import h5py
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
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
    "OQPSK"
]
def _normalize_data(data):
        sample = data # [2, 128]
        sample_min = sample.min()
        sample_max = sample.max()
        sample_range = sample_max - sample_min
        normalized = 2 * (sample - sample_min) / (sample_range + 1e-8) - 1
        return normalized
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

def _load_data(dataPath, classes, N_SNR):
    data =  []
    with h5py.File(dataPath, 'r') as f:
        for id in classes:
            # List all groups
            print("Keys: %s" % f.keys())
            # each snr values has 106496 total samples, so this pulls out the 4 highest snr values, we can just look
            data = f['X'][(106496*(id+1) - 4096*N_SNR):106496*(id+1)]
            # class_label = f['Y'][(106496*(id+1) - 4096*N_SNR):106496*(id+1)]
            # snr_value = f['Z'][(106496*(id+1) - 4096*N_SNR):106496*(id+1)]
            # data[(class_label[0],snr_value[0])] = data_slice
            # X is num_signals,Data, I/Q
    return data
# Load in the data
dataPath = '/home/hshayde/Projects/MIT/AMR/Dataset/2018.01/2018_RFML.hdf5'
selected_classes = ['16QAM']
data_classes  = [classes.index(cls) for cls in selected_classes]
snr_value = 1
data = _load_data(dataPath, data_classes, snr_value)
print('Data Loaded Successfully')
print(f'Data Shape: {data.shape}')
sample = data[0].T
sample = _normalize_data(sample)
print(sample[0])
import matplotlib.pyplot as plt

# Create subplots: constellation diagram and I/Q time series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Extract I and Q channels for each sample
sample_complex = sample[0] + 1j * sample[1]

# Plot constellation diagram
ax1.scatter(sample_complex.real, sample_complex.imag, alpha=0.6, s=20)
ax1.set_xlabel('In-Phase (I)')
ax1.set_ylabel('Quadrature (Q)')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.set_title('Constellation Diagram')

# Plot I and Q time series
time_samples = range(len(sample[0]))
ax2.plot(time_samples, sample[0], label='I (In-Phase)', alpha=0.8)
ax2.plot(time_samples, sample[1], label='Q (Quadrature)', alpha=0.8)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Amplitude')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('I and Q Time Series')

plt.tight_layout()
plt.show()



def mm_timing_sync(samples, sps=8, gain=0.3):
    """
    Mueller and Müller timing recovery for QPSK signals,
    operating on an upsampled-by-16 array.

    Args:
        samples: complex NumPy array, already upsampled (e.g. 16x)
        sps: original samples-per-symbol (at 1x). E.g. 8.
        gain: loop gain for timing adjustment

    Returns:
        synced: complex NumPy array of symbol-aligned samples
    """
    mu = 0.0
    N = len(samples)
    # Reserve a few extra entries for the delay pipeline
    out = np.zeros(N + 10, dtype=np.complex64)
    out_rail = np.zeros_like(out)

    i_in = 0        # pointer in upsampled domain
    i_out = 2       # output index (start at 2 for delay lines)
    sps_up = sps * 16  # upsampled samples per symbol

    while (i_out < N) and (i_in + sps_up < N):
        # fractional index in upsampled array
        idx = int(i_in + mu)
        frac = mu - int(mu)

        # linear interpolation between samples[idx] and samples[idx+1]
        s0 = samples[idx]
        s1 = samples[idx + 1]
        sample = s0 * (1 - frac) + s1 * frac

        out[i_out] = sample
        # hard decision to nearest QPSK quadrant
        out_rail[i_out] = (int(sample.real > 0) +
                           1j * int(sample.imag > 0))

        # Mueller-Müller timing error detector
        x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
        y = (out[i_out] - out[i_out - 2])       * np.conj(out_rail[i_out - 1])
        mm_val = np.real(y - x)

        # advance mu by one symbol (in upsampled samples) plus correction
        mu += sps_up + gain * mm_val
        # move integer part of mu into i_in
        i_in += int(mu)
        # keep only fractional remainder
        mu = mu - int(mu)
        i_out += 1

    # discard initial placeholders and any unused tail
    return out[2:i_out]

def costas_loop(signal, loop_bandwidth=0.05, damping_factor=0.707, modulation_order=64):
    """
    Costas Loop for carrier phase recovery in generic M-PSK signals.

    Args:
        signal (np.ndarray): Complex baseband input (after timing sync)
        loop_bandwidth (float): Determines loop responsiveness
        damping_factor (float): Controls damping in the loop filter
        modulation_order (int): e.g. 4 (QPSK), 8, 16, 18, etc.

    Returns:
        np.ndarray: Phase-corrected complex samples
    """
    N = len(signal)
    phase_est = 0.0
    freq_est = 0.0
    output = np.zeros(N, dtype=np.complex64)

    # Loop filter coefficients (2nd order PLL)
    theta = loop_bandwidth / (damping_factor + 0.25 / damping_factor)
    d = 1 + 2 * damping_factor * theta + theta**2
    alpha = (4 * damping_factor * theta) / d
    beta = (4 * theta**2) / d

    # PSK constellation (unit circle points)
    constellation = np.exp(1j * 2 * np.pi * np.arange(modulation_order) / modulation_order)

    for n in range(N):
        # Rotate by negative estimated phase
        corrected = signal[n] * np.exp(-1j * phase_est)
        output[n] = corrected

        # Find closest constellation point
        nearest = constellation[np.argmin(np.abs(corrected - constellation))]

        # Phase error (angle between received and ideal symbol)
        error = np.angle(corrected * np.conj(nearest))

        # Loop filter update
        freq_est += beta * error
        phase_est += freq_est + alpha * error

    return output
def dd_pll(signal, loop_bandwidth=0.005, damping_factor=.707, modulation_order=16):
    """
    Decision-Directed PLL for QAM signals.
    Suitable replacement for Costas Loop when amplitude varies (e.g., QAM16, QAM64).

    Args:
        signal (np.ndarray): Input complex signal (1D).
        loop_bandwidth (float): Loop bandwidth (small value like 0.005 or 0.0025).
        damping_factor (float): Damping factor, typical 0.707.
        modulation_order (int): QAM order (e.g., 16, 64).

    Returns:
        corrected_signal: Phase-corrected signal
        final_freq_offset: Estimated frequency offset (normalized cycles/sample)
        final_phase_offset: Estimated phase offset (radians)
    """
    N = len(signal)
    phase_est = 0.0
    freq_est = 0.0
    output = np.zeros(N, dtype=np.complex64)

    # Loop filter coefficients (same logic as Costas)
    theta = loop_bandwidth / (damping_factor + 0.25 / damping_factor)
    d = 1 + 2 * damping_factor * theta + theta**2
    alpha = (4 * damping_factor * theta) / d
    beta = (4 * theta**2) / d

    # Generate QAM constellation
    m = int(np.sqrt(modulation_order))
    real_levels = np.arange(-(m - 1), m + 1, 2)
    imag_levels = np.arange(-(m - 1), m + 1, 2)
    constellation = np.array([r + 1j * i for r in real_levels for i in imag_levels])

    # Normalize constellation energy to 1
    constellation /= np.sqrt((np.abs(constellation)**2).mean())

    for n in range(N):
        corrected = signal[n] * np.exp(-1j * phase_est)
        output[n] = corrected

        # Find nearest QAM symbol
        nearest = constellation[np.argmin(np.abs(corrected - constellation))]
        error = np.angle(corrected * np.conj(nearest))

        freq_est += beta * error
        phase_est += freq_est + alpha * error

    final_phase_offset = (phase_est % (2 * np.pi))
    if final_phase_offset > np.pi:
        final_phase_offset -= 2 * np.pi
    final_freq_offset = freq_est / (2 * np.pi)

    return output, final_freq_offset, final_phase_offset
# Now we can see that we are not time synces, lets play around with pySDR to lock in these time syncrhinizations first and then go from there
from scipy import signal
# Oversample (interpolate) by 16x
sample = data[0].T
sample = _normalize_data(sample)
sample_complex = sample[0] + 1j * sample[1]
samples_interpolated = signal.resample_poly(sample_complex, up=16, down=1)
synced_qpsk = mm_timing_sync(samples_interpolated, sps=8)
# synced_qpsk = costas_loop(synced_qpsk)
synced_qpsk, _, _ = dd_pll(synced_qpsk)
# synced_qpsk = signal.resample(synced_qpsk, 1024)

# Create side-by-side comparison plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot original signal (not interpolated/synced)
ax1.plot(sample_complex.real[::16], sample_complex.imag[::16], '.', alpha=0.5)
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
