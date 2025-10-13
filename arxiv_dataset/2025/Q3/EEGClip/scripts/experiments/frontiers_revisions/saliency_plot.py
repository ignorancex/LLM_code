import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.channels import make_standard_montage

def plot_1d_average_gradients(amp_grads, figsize=(12, 8), fontsize=18):
    """
    Plot 1D average gradients over all electrodes
    
    Parameters:
    amp_grads: numpy array, shape (n_samples, n_channels, n_frequencies)
        Amplitude gradients data
    figsize: tuple, figure size (width, height)
    fontsize: int, base font size for labels and titles
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    INPUT_WINDOW_SAMPLES = 1200  
    SFREQ = 100  
    
    # Calculate frequency array
    freqs = np.fft.rfftfreq(INPUT_WINDOW_SAMPLES, 1/SFREQ)
    
    # Calculate overall frequency-channel gradients
    amp_grads_mean = np.mean(amp_grads, axis=0)  # Average across samples
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Average over channels for 1D plot
    avg_over_channels = np.mean(amp_grads_mean, axis=0)
    
    # Plot with thicker line for better visibility
    ax.plot(freqs, avg_over_channels, 'b-', linewidth=4)
    ax.set_xlabel('Frequency (Hz)', fontsize=fontsize)
    ax.set_ylabel('Average Gradient', fontsize=fontsize)
    ax.set_title('Average Gradient Across All Electrodes', fontsize=fontsize+4, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_xlim(freqs[0], freqs[-1])
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    # Make spines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    return fig, ax

def plot_frequency_band_topography(amp_grads, figsize=(18, 12), fontsize=18, show=True):
    """
    Plot EEG scalp topography for different frequency bands
    
    Parameters:
    amp_grads: numpy array, shape (n_samples, n_channels, n_frequencies)
        Amplitude gradients data
    figsize: tuple, figure size (width, height)
    fontsize: int, base font size for labels and titles
    show: bool, whether to display the plot (default: True)
    
    Returns:
    fig: matplotlib figure object
    """
    INPUT_WINDOW_SAMPLES = 1200  
    SFREQ = 100  
    
    # Temporarily set matplotlib to non-interactive mode to prevent automatic showing
    plt.ioff()
    
    # Channel information
    channel_names = ['EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 
                    'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 
                    'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 
                    'EEG O1-REF', 'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 
                    'EEG PZ-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 
                    'EEG T6-REF']
    
    # Clean channel names
    clean_names = [name.replace('EEG ', '').replace('-REF', '') for name in channel_names]
    
    # Channel mapping to standard montage names
    channel_mapping = {
        'A1': 'A1', 'A2': 'A2', 'C3': 'C3', 'C4': 'C4',
        'CZ': 'Cz', 'F3': 'F3', 'F4': 'F4', 'F7': 'F7',
        'F8': 'F8', 'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz',
        'O1': 'O1', 'O2': 'O2', 'P3': 'P3', 'P4': 'P4',
        'PZ': 'Pz', 'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
    }
    
    # Map channel names to standard montage names
    standard_names = [channel_mapping[name] for name in clean_names]
    
    # Calculate frequency array
    freqs = np.fft.rfftfreq(INPUT_WINDOW_SAMPLES, 1/SFREQ)
    
    # Define frequency bands
    freq_bands = {
        'Delta (0-4 Hz)': (0, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-14 Hz)': (8, 14),
        'Beta (14-20 Hz)': (14, 20),
        'Low Gamma (20-30 Hz)': (20, 30),
        'High Gamma (30-50 Hz)': (30, 50)
    }
    
    # Function to get frequency indices for a given band
    def get_freq_indices(freq_min, freq_max):
        return np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    
    # Create MNE info structure
    info = mne.create_info(ch_names=standard_names, sfreq=100, ch_types='eeg')
    montage = make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Find global min/max for consistent color scaling across all topoplots
    all_band_data = []
    for band_name, (freq_min, freq_max) in freq_bands.items():
        freq_indices = get_freq_indices(freq_min, freq_max)
        if len(freq_indices) > 0:
            band_data = np.mean(amp_grads[:, :, freq_indices], axis=(0, 2))
            all_band_data.append(band_data)
    
    if all_band_data:
        vmin = np.min(all_band_data)
        vmax = np.max(all_band_data)
    else:
        vmin, vmax = None, None
    
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Create scalp plots for each frequency band
    for i, (band_name, (freq_min, freq_max)) in enumerate(freq_bands.items()):
        ax = axes[i]
        
        # Get frequency indices for this band
        freq_indices = get_freq_indices(freq_min, freq_max)
        
        if len(freq_indices) > 0:
            # Calculate average gradient for this frequency band
            band_data = np.mean(amp_grads[:, :, freq_indices], axis=(0, 2))
            
            # Create evoked object for this band
            evoked_band = mne.EvokedArray(band_data.reshape(-1, 1), info)
            
            # Plot scalp topography with increased contour lines and better styling
            im, cn = mne.viz.plot_topomap(band_data, evoked_band.info, 
                                         axes=ax, 
                                         show=False,  # Crucial: prevent automatic showing
                                         contours=8,  # More contour lines
                                         cmap='RdBu_r',
                                         vlim=(vmin, vmax),
                                         sensors=True,  # Show electrode positions
                                         names=None,    # Don't show electrode names (too cluttered)
                                         mask_params=dict(marker='o', markerfacecolor='w', 
                                                        markeredgecolor='k', linewidth=0, markersize=6))
            ax.set_title(band_name, fontsize=fontsize, fontweight='bold', pad=25)
        else:
            ax.text(0.5, 0.5, f'No data\nfor {band_name}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=fontsize-2)
            ax.set_title(band_name, fontsize=fontsize, fontweight='bold', pad=25)
    
    # Add main title with more space
    fig.suptitle('EEG Topography by Frequency Bands', fontsize=fontsize+6, fontweight='bold', y=0.95)
    
    # Adjust layout with more space at the top and right
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.82, hspace=0.35, wspace=0.25)  # More space for title and colorbar
    
    # Add a single colorbar for all topoplots
    if all_band_data:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap='RdBu_r')
        sm.set_array([])
        
        # Create colorbar axes manually for better control
        cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Average Gradient', rotation=270, labelpad=25, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Restore interactive mode and show if requested
    plt.ion()
    
    return fig

def plot_gradients_and_topography(amp_grads):
    """
    Plot frequency-channel gradients and EEG scalp topography for different frequency bands
    
    Parameters:
    amp_grads: numpy array, shape (n_samples, n_channels, n_frequencies)
        Amplitude gradients data
    """
    INPUT_WINDOW_SAMPLES = 1200  
    SFREQ = 100  
    
    # Channel information
    channel_names = ['EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 
                    'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 
                    'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 
                    'EEG O1-REF', 'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 
                    'EEG PZ-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 
                    'EEG T6-REF']
    
    # Clean channel names
    clean_names = [name.replace('EEG ', '').replace('-REF', '') for name in channel_names]
    
    # Channel mapping to standard montage names
    channel_mapping = {
        'A1': 'A1', 'A2': 'A2', 'C3': 'C3', 'C4': 'C4',
        'CZ': 'Cz', 'F3': 'F3', 'F4': 'F4', 'F7': 'F7',
        'F8': 'F8', 'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz',
        'O1': 'O1', 'O2': 'O2', 'P3': 'P3', 'P4': 'P4',
        'PZ': 'Pz', 'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
    }
    
    # Map channel names to standard montage names
    standard_names = [channel_mapping[name] for name in clean_names]
    
    # Calculate frequency array
    freqs = np.fft.rfftfreq(INPUT_WINDOW_SAMPLES, 1/SFREQ)
    
    # Define frequency bands
    freq_bands = {
        'Delta (0-4 Hz)': (0, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-14 Hz)': (8, 14),
        'Beta (14-20 Hz)': (14, 20),
        'Low Gamma (20-30 Hz)': (20, 30),
        'High Gamma (30-50 Hz)': (30, 50)
    }
    
    # Function to get frequency indices for a given band
    def get_freq_indices(freq_min, freq_max):
        return np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    
    # Create MNE info structure
    info = mne.create_info(ch_names=standard_names, sfreq=100, ch_types='eeg')
    montage = make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Calculate overall frequency-channel gradients for the left plot
    amp_grads_mean = np.mean(amp_grads, axis=0)  # Average across samples
    #amp_grads_mean = np.expand_dims(amp_grads_mean, axis=0)
    
    # Create figure with frequency-channel gradient plot and frequency band scalp plots
    fig = plt.figure(figsize=(20, 12))
    
    # Left top plot: 2D Frequency-Channel Gradients
    ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=1)
    im1 = ax1.imshow(amp_grads_mean, aspect='auto', cmap='RdBu_r', 
                     extent=[freqs[0], freqs[-1], 0, amp_grads.shape[1]])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Channel')
    ax1.set_title('2D: Frequency-Channel Gradients', fontsize=12)
    
    # Add channel labels to y-axis
    ax1.set_yticks(range(len(clean_names)))
    ax1.set_yticklabels(clean_names, fontsize=8)
    
    # Add colorbar for 2D plot
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Gradient Magnitude', rotation=270, labelpad=15)
    
    # Left bottom plot: 1D Average over electrodes
    ax2 = plt.subplot2grid((3, 4), (2, 0), colspan=1)
    avg_over_channels = np.mean(amp_grads_mean, axis=0)  # Average over channels
    ax2.plot(freqs, avg_over_channels, 'b-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Average Gradient')
    ax2.set_title('1D: Average over Electrodes', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])
    
    # Right side: Create scalp plots for each frequency band
    positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    
    # Find global min/max for consistent color scaling across all topoplots
    all_band_data = []
    for band_name, (freq_min, freq_max) in freq_bands.items():
        freq_indices = get_freq_indices(freq_min, freq_max)
        if len(freq_indices) > 0:
            band_data = np.mean(amp_grads[:, :, freq_indices], axis=(0, 2))  # Average across samples and frequencies in band
            all_band_data.append(band_data)
    
    if all_band_data:
        vmin = np.min(all_band_data)
        vmax = np.max(all_band_data)
    else:
        vmin, vmax = None, None
    
    # Create scalp plots for each frequency band
    for i, (band_name, (freq_min, freq_max)) in enumerate(freq_bands.items()):
        if i < len(positions):
            row, col = positions[i]
            if i < 3:  # First row
                ax = plt.subplot2grid((3, 4), (row, col))
            else:  # Second row
                ax = plt.subplot2grid((3, 4), (row, col))
            
            # Get frequency indices for this band
            freq_indices = get_freq_indices(freq_min, freq_max)
            
            if len(freq_indices) > 0:
                # Calculate average gradient for this frequency band
                band_data = np.mean(amp_grads[:, :, freq_indices], axis=(0, 2))  # Average across samples and frequencies in band
                
                # Create evoked object for this band
                evoked_band = mne.EvokedArray(band_data.reshape(-1, 1), info)
                
                # Plot scalp topography
                im, cn = mne.viz.plot_topomap(band_data, evoked_band.info, 
                                             axes=ax, 
                                             show=False,
                                             contours=6,
                                             cmap='RdBu_r',
                                             vlim=(vmin, vmax))
                ax.set_title(band_name, fontsize=12)
            else:
                ax.text(0.5, 0.5, f'No data\nfor {band_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(band_name, fontsize=12)
    
    # Add a single colorbar for all topoplots
    if all_band_data:
        # Create a dummy mappable for the colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap='RdBu_r')
        sm.set_array([])
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar2 = plt.colorbar(sm, cax=cbar_ax)
        cbar2.set_label('Average Gradient', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)

    
    #return fig

# Example usage:
# fig = plot_gradients_and_topography(your_amp_grads_data)