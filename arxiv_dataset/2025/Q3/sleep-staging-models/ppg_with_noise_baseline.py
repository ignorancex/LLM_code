"""
ppg_with_noise_baseline.py
Single-stream PPG + Noise baseline model
Used to verify if performance improvement comes from noise or dual-stream architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multimodal_sleep_model import SleepPPGNet


class PPGWithNoiseBaseline(nn.Module):
    """
    Single-stream PPG + Noise model
    Directly uses SleepPPG-Net architecture but with noise-augmented PPG input
    """

    def __init__(self, noise_config=None):
        super().__init__()

        # Reuse SleepPPG-Net architecture
        self.model = SleepPPGNet()

        # Noise configuration (consistent with dual-stream model)
        self.noise_config = noise_config or {
            'noise_level': 0.1,  # Gaussian noise standard deviation
            'drift_amplitude': 0.1,  # Baseline drift amplitude
            'drift_frequency': 0.1,  # Baseline drift frequency
            'spike_probability': 0.01,  # Motion artifact probability
            'spike_amplitude': 0.5  # Motion artifact amplitude
        }

    def add_noise_to_ppg(self, clean_ppg):
        """
        Add noise to clean PPG signal (exact same implementation as dual-stream model)

        Args:
            clean_ppg: Clean PPG signal (B, 1, L)
        Returns:
            noisy_ppg: Noisy PPG signal (B, 1, L)
        """
        batch_size, _, length = clean_ppg.shape
        device = clean_ppg.device

        # Copy signal
        noisy_ppg = clean_ppg.clone()

        # 1. Add Gaussian white noise
        gaussian_noise = torch.randn_like(clean_ppg) * self.noise_config['noise_level']
        noisy_ppg = noisy_ppg + gaussian_noise

        # 2. Add baseline drift (low-frequency noise)
        t = torch.linspace(0, 1, length, device=device)
        drift_freq = self.noise_config['drift_frequency']
        drift_amp = self.noise_config['drift_amplitude']

        # Combination of multiple low-frequency components
        drift = drift_amp * (
                0.5 * torch.sin(2 * np.pi * drift_freq * t) +
                0.3 * torch.sin(2 * np.pi * drift_freq * 2 * t) +
                0.2 * torch.sin(2 * np.pi * drift_freq * 0.5 * t)
        )
        drift = drift.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        noisy_ppg = noisy_ppg + drift

        # 3. Add motion artifacts (random spikes)
        spike_prob = self.noise_config['spike_probability']
        spike_amp = self.noise_config['spike_amplitude']

        # Generate random spike locations
        spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
        spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
        spikes = spike_mask.float() * spike_values

        # Smooth spikes (make them more realistic)
        kernel_size = 5
        padding = kernel_size // 2
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)

        noisy_ppg = noisy_ppg + spikes

        # 4. Add high-frequency oscillation (EMG interference)
        emg_noise = torch.randn_like(clean_ppg) * 0.05
        noisy_ppg = noisy_ppg + emg_noise

        return noisy_ppg

    def forward(self, ppg):
        """
        Forward pass

        Args:
            ppg: Clean PPG signal (B, 1, 1228800)
        Returns:
            output: Sleep stage predictions (B, 4, 1200)
        """
        # Add noise
        noisy_ppg = self.add_noise_to_ppg(ppg)

        # Pass through SleepPPG-Net
        output = self.model(noisy_ppg)

        return output


def test_model():
    """Test the model"""
    print("Testing PPG with Noise Baseline Model...")

    # Create model
    model = PPGWithNoiseBaseline()

    # Test input
    ppg = torch.randn(2, 1, 1228800)  # 10 hours of data

    # Test forward pass
    output = model(ppg)
    print(f"Input shape: {ppg.shape}")
    print(f"Output shape: {output.shape}")  # Should be (2, 4, 1200)

    # Parameter count (should be same as SleepPPG-Net)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify noise addition
    clean_ppg = torch.randn(1, 1, 1000)
    noisy_ppg = model.add_noise_to_ppg(clean_ppg)

    print(f"\nNoise statistics:")
    print(f"Clean PPG - mean: {clean_ppg.mean():.4f}, std: {clean_ppg.std():.4f}")
    print(f"Noisy PPG - mean: {noisy_ppg.mean():.4f}, std: {noisy_ppg.std():.4f}")
    print(f"Difference - std: {(noisy_ppg - clean_ppg).std():.4f}")


if __name__ == "__main__":
    test_model()