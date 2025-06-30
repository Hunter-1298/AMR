import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from collections import defaultdict

# [Previous imports and functions remain the same until after the synchronization code]

# Add these analysis functions after your existing code:

def extract_psk_features(synced_signal, mod_order, sample_rate=1.0):
    """Extract distinguishing features for PSK signals"""
    features = {}

    # 1. Symbol transitions analysis
    # Detect symbol boundaries and measure transition characteristics
    phase = np.angle(synced_signal)
    phase_diff = np.diff(np.unwrap(phase))

    # Expected phase steps for PSK
    expected_phase_step = 2 * np.pi / mod_order

    # Detect large phase transitions (symbol changes)
    threshold = expected_phase_step / 2
    transitions = np.abs(phase_diff) > threshold

    features['transition_rate'] = np.sum(transitions) / len(synced_signal)
    features['avg_phase_diff'] = np.mean(np.abs(phase_diff))
    features['std_phase_diff'] = np.std(phase_diff)

    # 2. Phase noise characteristics
    # Measure deviation from ideal constellation points
    ideal_phases = np.linspace(0, 2*np.pi, mod_order, endpoint=False)
    phase_errors = []

    for p in phase:
        # Find nearest ideal phase
        distances = np.abs(p - ideal_phases)
        min_dist = np.min(np.minimum(distances, 2*np.pi - distances))
        phase_errors.append(min_dist)

    features['mean_phase_error'] = np.mean(phase_errors)
    features['std_phase_error'] = np.std(phase_errors)
    features['max_phase_error'] = np.max(phase_errors)

    # 3. Amplitude variations
    amplitude = np.abs(synced_signal)
    features['amp_mean'] = np.mean(amplitude)
    features['amp_std'] = np.std(amplitude)
    features['amp_range'] = np.max(amplitude) - np.min(amplitude)

    # 4. Higher-order statistics
    features['kurtosis_real'] = stats.kurtosis(synced_signal.real)
    features['kurtosis_imag'] = stats.kurtosis(synced_signal.imag)
    features['skewness_abs'] = stats.skew(np.abs(synced_signal))

    # 5. Cyclostationary features
    # Autocorrelation at symbol period
    if len(synced_signal) > 100:
        autocorr = np.correlate(synced_signal, synced_signal, mode='same')
        autocorr = autocorr / np.max(np.abs(autocorr))
        features['autocorr_peak2'] = np.abs(autocorr[len(autocorr)//2 + 8])  # At symbol period

    # 6. Spectral features
    spectrum = np.abs(fft(synced_signal))[:len(synced_signal)//2]
    features['spectral_peak'] = np.max(spectrum)
    features['spectral_bandwidth'] = np.sum(spectrum > 0.1 * np.max(spectrum))

    return features

def plot_psk_analysis(all_data, modulation_orders):
    """Comprehensive PSK analysis and visualization"""

    # Process multiple samples per modulation type
    features_by_mod = defaultdict(list)
    signals_by_mod = defaultdict(list)

    for mod_type, data in all_data.items():
        mod_order = modulation_orders[mod_type]
        print(f'\nAnalyzing {mod_type} (order={mod_order})...')

        # Process multiple samples
        num_samples = min(50, len(data))  # Analyze up to 50 samples

        for i in range(num_samples):
            sample = data[i].T
            sample = _normalize_data(sample)
            sample_complex = sample[0] + 1j * sample[1]

            # Synchronize
            samples_interpolated = signal.resample_poly(sample_complex, up=16, down=1)
            synced = mm_timing_sync(samples_interpolated, sps=8, modulation_order=mod_order)
            synced = costas_loop(synced, modulation_order=mod_order)

            # Extract features
            features = extract_psk_features(synced, mod_order)
            features_by_mod[mod_type].append(features)
            signals_by_mod[mod_type].append(synced[:1000])  # Store first 1000 samples

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))

    # 1. Phase transition patterns
    ax1 = plt.subplot(4, 4, 1)
    for mod_type in selected_classes:
        # Get phase differences for first signal
        signal_sample = signals_by_mod[mod_type][0]
        phase = np.angle(signal_sample)
        phase_diff = np.diff(np.unwrap(phase))

        ax1.hist(phase_diff, bins=50, alpha=0.5, label=mod_type, density=True)

    ax1.set_title('Phase Difference Distribution')
    ax1.set_xlabel('Phase Difference (radians)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Symbol transition rate analysis
    ax2 = plt.subplot(4, 4, 2)
    transition_rates = {mod: [f['transition_rate'] for f in features_by_mod[mod]]
                       for mod in selected_classes}

    positions = np.arange(len(selected_classes))
    bp = ax2.boxplot([transition_rates[mod] for mod in selected_classes],
                     positions=positions, patch_artist=True)
    ax2.set_xticklabels(selected_classes)
    ax2.set_title('Symbol Transition Rates')
    ax2.set_ylabel('Transition Rate')
    ax2.grid(True, alpha=0.3)

    # Color the boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_classes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 3. Phase error analysis
    ax3 = plt.subplot(4, 4, 3)
    phase_errors = {mod: [f['mean_phase_error'] for f in features_by_mod[mod]]
                   for mod in selected_classes}

    for i, mod in enumerate(selected_classes):
        ax3.scatter([i]*len(phase_errors[mod]), phase_errors[mod],
                   alpha=0.5, s=30, label=mod)

    ax3.set_xticks(range(len(selected_classes)))
    ax3.set_xticklabels(selected_classes)
    ax3.set_title('Phase Error Distribution')
    ax3.set_ylabel('Mean Phase Error (radians)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Time-domain symbol patterns
    ax4 = plt.subplot(4, 4, 4)
    time_window = 100
    for mod_type in selected_classes:
        signal_sample = signals_by_mod[mod_type][0][:time_window]
        ax4.plot(np.abs(signal_sample), label=f'{mod_type} magnitude', alpha=0.7)

    ax4.set_title('Time-Domain Magnitude')
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Eye diagram (I channel)
    for idx, mod_type in enumerate(selected_classes):
        ax = plt.subplot(4, 4, 5 + idx)
        signal_sample = signals_by_mod[mod_type][0]

        # Create eye diagram
        samples_per_symbol = 16  # After interpolation
        num_symbols = len(signal_sample) // samples_per_symbol

        for i in range(min(num_symbols-2, 100)):  # Limit traces for clarity
            start = i * samples_per_symbol
            end = start + 2 * samples_per_symbol
            if end < len(signal_sample):
                ax.plot(signal_sample.real[start:end], alpha=0.3, color='blue')

        ax.set_title(f'{mod_type} Eye Diagram (I)')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

    # 6. Power Spectral Density
    ax9 = plt.subplot(4, 4, 9)
    for mod_type in selected_classes:
        signal_sample = signals_by_mod[mod_type][0]
        freqs, psd = signal.welch(signal_sample, nperseg=256)
        ax9.semilogy(freqs, np.abs(psd), label=mod_type, alpha=0.7)

    ax9.set_title('Power Spectral Density')
    ax9.set_xlabel('Normalized Frequency')
    ax9.set_ylabel('PSD')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 7. Feature correlation matrix
    ax10 = plt.subplot(4, 4, 10)

    # Collect all features into matrix
    feature_names = list(features_by_mod[selected_classes[0]][0].keys())
    feature_matrix = []
    labels = []

    for mod_type in selected_classes:
        for features in features_by_mod[mod_type]:
            feature_matrix.append([features[fname] for fname in feature_names])
            labels.append(mod_type)

    feature_matrix = np.array(feature_matrix)

    # Normalize features
    feature_matrix_norm = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)

    # Compute correlation
    corr_matrix = np.corrcoef(feature_matrix_norm.T)

    im = ax10.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax10.set_xticks(range(len(feature_names)))
    ax10.set_yticks(range(len(feature_names)))
    ax10.set_xticklabels(feature_names, rotation=45, ha='right')
    ax10.set_yticklabels(feature_names)
    ax10.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax10)

    # 8. t-SNE visualization of features
    ax11 = plt.subplot(4, 4, 11)

    if len(feature_matrix) > 3:
        tsne = TSNE(n_components=2, perplexity=min(30, len(feature_matrix)-1))
        features_2d = tsne.fit_transform(feature_matrix_norm)

        for mod_type in selected_classes:
            mask = np.array(labels) == mod_type
            ax11.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        label=mod_type, alpha=0.6, s=50)

        ax11.set_title('t-SNE of PSK Features')
        ax11.set_xlabel('t-SNE 1')
        ax11.set_ylabel('t-SNE 2')
        ax11.legend()
        ax11.grid(True, alpha=0.3)

    # 9. Histogram of amplitude variations
    ax12 = plt.subplot(4, 4, 12)
    for mod_type in selected_classes:
        amp_stds = [f['amp_std'] for f in features_by_mod[mod_type]]
        ax12.hist(amp_stds, bins=20, alpha=0.5, label=mod_type, density=True)

    ax12.set_title('Amplitude Variation Distribution')
    ax12.set_xlabel('Amplitude Std Dev')
    ax12.set_ylabel('Density')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    # 10. Cyclostationary features
    ax13 = plt.subplot(4, 4, 13)
    for mod_type in selected_classes:
        signal_sample = signals_by_mod[mod_type][0]

        # Compute cyclic autocorrelation
        lags = np.arange(-50, 51)
        autocorr = np.correlate(signal_sample[:500], signal_sample[:500], mode='same')
        center = len(autocorr) // 2

        ax13.plot(lags, np.abs(autocorr[center-50:center+51]),
                 label=mod_type, alpha=0.7)

    ax13.set_title('Cyclic Autocorrelation')
    ax13.set_xlabel('Lag')
    ax13.set_ylabel('|Autocorrelation|')
    ax13.legend()
    ax13.grid(True, alpha=0.3)

    # 11. Phase clustering quality
    ax14 = plt.subplot(4, 4, 14)
    clustering_quality = {}

    for mod_type in selected_classes:
        mod_order = modulation_orders[mod_type]
        signal_sample = signals_by_mod[mod_type][0]

        # Measure how well symbols cluster around ideal points
        phase = np.angle(signal_sample)
        ideal_phases = np.linspace(0, 2*np.pi, mod_order, endpoint=False)

        # Assign each sample to nearest ideal phase
        cluster_vars = []
        for ideal in ideal_phases:
            # Find samples near this ideal phase
            distances = np.abs(phase - ideal)
            distances = np.minimum(distances, 2*np.pi - distances)
            nearby = distances < (np.pi / mod_order)

            if np.sum(nearby) > 0:
                cluster_phases = phase[nearby]
                # Compute circular variance
                cluster_var = 1 - np.abs(np.mean(np.exp(1j * cluster_phases)))
                cluster_vars.append(cluster_var)

        clustering_quality[mod_type] = np.mean(cluster_vars) if cluster_vars else 0

    mods = list(clustering_quality.keys())
    qualities = list(clustering_quality.values())
    ax14.bar(mods, qualities, alpha=0.7)
    ax14.set_title('Phase Clustering Quality')
    ax14.set_ylabel('Mean Circular Variance')
    ax14.grid(True, alpha=0.3)

    # 12. Symbol rate estimation accuracy
    ax15 = plt.subplot(4, 4, 15)

    # Feature importance for your VQ-MAE
    feature_importance = {
        'transition_rate': [],
        'mean_phase_error': [],
        'amp_std': [],
        'kurtosis_real': [],
        'spectral_bandwidth': []
    }

    for mod_type in selected_classes:
        for feat_name in feature_importance.keys():
            values = [f[feat_name] for f in features_by_mod[mod_type]]
            feature_importance[feat_name].append(np.mean(values))

    # Plot as grouped bar chart
    x = np.arange(len(selected_classes))
    width = 0.15

    for i, (feat_name, values) in enumerate(feature_importance.items()):
        ax15.bar(x + i*width, values, width, label=feat_name, alpha=0.7)

    ax15.set_xlabel('Modulation Type')
    ax15.set_ylabel('Feature Value (normalized)')
    ax15.set_title('Key Discriminative Features')
    ax15.set_xticks(x + width * 2)
    ax15.set_xticklabels(selected_classes)
    ax15.legend(loc='upper left', fontsize=8)
    ax15.grid(True, alpha=0.3)

    # 13. Print feature statistics
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('off')

    stats_text = "Feature Statistics for VQ-MAE:\n\n"

    # Find most discriminative features
    feature_stds = {}
    for feat_name in feature_names:
        values_by_mod = []
        for mod_type in selected_classes:
            mod_values = [f[feat_name] for f in features_by_mod[mod_type]]
            values_by_mod.append(np.mean(mod_values))

        # Coefficient of variation across modulation types
        feature_stds[feat_name] = np.std(values_by_mod) / (np.mean(values_by_mod) + 1e-8)

    # Sort by discriminative power
    sorted_features = sorted(feature_stds.items(), key=lambda x: x[1], reverse=True)

    stats_text += "Most Discriminative Features:\n"
    for feat, score in sorted_features[:5]:
        stats_text += f"  {feat}: {score:.3f}\n"

    stats_text += "\nRecommendations for VQ-MAE:\n"
    stats_text += "1. Use phase transition patterns\n"
    stats_text += "2. Include amplitude variation features\n"
    stats_text += "3. Consider cyclostationary properties\n"
    stats_text += "4. Hierarchical: PSK order groups\n"

    ax16.text(0.05, 0.95, stats_text, transform=ax16.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)

    plt.tight_layout()
    plt.show()

    return features_by_mod

# Update the main execution code
# Load in the data
dataPath = '/home/hshayde/Projects/MIT/AMR/Dataset/2018.01/2018_RFML.hdf5'

# Select four PSK modulation types
selected_classes = ["QPSK", "8PSK", "16PSK", "32PSK"]
modulation_orders = {"QPSK": 4, "8PSK": 8, "16PSK": 16, "32PSK": 32}

data_classes = [classes.index(cls) for cls in selected_classes]
snr_value = 1

# Load data for all selected classes
all_data = _load_data(dataPath, data_classes, snr_value)
print('Data Loaded Successfully')

# Run comprehensive PSK analysis
features_by_mod = plot_psk_analysis(all_data, modulation_orders)

# Additional analysis for hierarchical VQ-MAE design
print("\n=== Hierarchical VQ-MAE Design Suggestions ===")
print("\n1. First Level - PSK Order Groups:")
print("   - Low order: QPSK")
print("   - Medium order: 8PSK")
print("   - High order: 16PSK, 32PSK")

print("\n2. Key Features to Encode:")
for mod_type in selected_classes:
    features = features_by_mod[mod_type][0]  # First sample
    print(f"\n{mod_type}:")
    print(f"  - Transition rate: {features['transition_rate']:.3f}")
    print(f"  - Phase error: {features['mean_phase_error']:.3f}")
    print(f"  - Amplitude std: {features['amp_std']:.3f}")
