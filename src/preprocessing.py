import numpy as np
from scipy.signal import butter, filtfilt, detrend

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(signal, fs=500, lowcut=0.5, highcut=50.0, order=5):
    """
    Apply Butterworth bandpass filter to ECG signal.
    Args:
        signal (np.array): ECG signal of shape (leads, length) or (length,)
        fs (int): Sampling frequency
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        order (int): Filter order
    Returns:
        np.array: Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply filter along the time axis (last axis)
    filtered_signal = filtfilt(b, a, signal, axis=-1)
    return filtered_signal

def normalize(signal):
    """
    Z-score normalization per lead.
    Args:
        signal (np.array): ECG signal of shape (leads, length)
    Returns:
        np.array: Normalized signal
    """
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return (signal - mean) / std

def augment(signal, noise_level=0.01, stretch_factor=1.0):
    """
    Apply random augmentation to the ECG signal.
    Args:
        signal (np.array): ECG signal
        noise_level (float): Std dev of Gaussian noise to add
        stretch_factor (float): Factor to stretch/squeeze signal (implementation placeholder)
    Returns:
        np.array: Augmented signal
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, signal.shape)
    augmented_signal = signal + noise
    
    # Note: Stretching requires resampling which can be computationally expensive on-the-fly.
    # For now, we stick to noise injection.
    
    return augmented_signal
