"""
Data Augmentation for ECG Signals
"""
import torch
import numpy as np

class ECGAugmentation:
    """Collection of augmentation techniques for ECG signals"""
    
    @staticmethod
    def time_warp(signal, sigma=0.2):
        """
        Apply time warping to ECG signal
        Args:
            signal: (12, 5000) ECG tensor
            sigma: Warping strength
        Returns:
            Warped signal
        """
        if isinstance(signal, torch.Tensor):
            device = signal.device
            signal_np = signal.cpu().numpy()
        else:
            device = 'cpu'
            signal_np = signal
            
        num_leads, seq_len = signal_np.shape
        
        # Create smooth random warp
        warp = np.cumsum(np.random.randn(seq_len) * sigma)
        warp = warp - warp.min()
        warp = warp / warp.max() * (seq_len - 1)
        warp = warp.astype(int)
        
        # Apply warp to each lead
        warped = np.zeros_like(signal_np)
        for i in range(num_leads):
            warped[i] = signal_np[i, warp]
        
        return torch.tensor(warped, dtype=torch.float32, device=device)
    
    @staticmethod
    def add_gaussian_noise(signal, noise_level=0.05):
        """
        Add Gaussian noise to ECG signal
        Args:
            signal: (12, 5000) ECG tensor
            noise_level: Standard deviation of noise
        Returns:
            Noisy signal
        """
        noise = torch.randn_like(signal) * noise_level
        return signal + noise
    
    @staticmethod
    def scale_amplitude(signal, scale_range=(0.8, 1.2)):
        """
        Randomly scale signal amplitude
        Args:
            signal: (12, 5000) ECG tensor
            scale_range: (min, max) scaling factors
        Returns:
            Scaled signal
        """
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def mixup(signal1, signal2, labels1, labels2, alpha=0.2):
        """
        Mixup augmentation
        Args:
            signal1, signal2: ECG signals
            labels1, labels2: Corresponding labels
            alpha: Mixup parameter
        Returns:
            Mixed signal and labels
        """
        lam = np.random.beta(alpha, alpha)
        mixed_signal = lam * signal1 + (1 - lam) * signal2
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        return mixed_signal, mixed_labels
    
    @staticmethod
    def random_crop_and_pad(signal, crop_ratio=0.9):
        """
        Randomly crop and pad signal
        Args:
            signal: (12, 5000) ECG tensor
            crop_ratio: Ratio of signal to keep
        Returns:
            Cropped and padded signal
        """
        num_leads, seq_len = signal.shape
        crop_len = int(seq_len * crop_ratio)
        
        start = np.random.randint(0, seq_len - crop_len)
        cropped = signal[:, start:start+crop_len]
        
        # Pad to original length
        pad_left = (seq_len - crop_len) // 2
        pad_right = seq_len - crop_len - pad_left
        
        padded = torch.nn.functional.pad(cropped, (pad_left, pad_right), mode='constant', value=0)
        return padded
    
    @staticmethod
    def apply_random_augmentation(signal, p=0.5):
        """
        Apply random augmentation with probability p
        Args:
            signal: (12, 5000) ECG tensor
            p: Probability of applying augmentation
        Returns:
            Augmented signal
        """
        if np.random.rand() < p:
            aug_type = np.random.choice(['noise', 'scale', 'warp', 'crop'])
            
            if aug_type == 'noise':
                return ECGAugmentation.add_gaussian_noise(signal)
            elif aug_type == 'scale':
                return ECGAugmentation.scale_amplitude(signal)
            elif aug_type == 'warp':
                return ECGAugmentation.time_warp(signal)
            elif aug_type == 'crop':
                return ECGAugmentation.random_crop_and_pad(signal)
        
        return signal
