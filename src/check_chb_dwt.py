import numpy as np, pywt, warnings
warnings.simplefilter("ignore")

# Same layout the authors build for CHB-MIT: (windows, 18 channels, 1024 samples, 1)
eeg_data = np.random.randn(5, 18, 1024, 1).astype(np.float32)

# What extract_coeffs_BONN_CHB does for one window:
signal = eeg_data[0, :, 0]
print("signal shape:", signal.shape)

coeffs = pywt.wavedec(signal, 'db1', level=3)
print("coefficient shapes:", [c.shape for c in coeffs])
print("concatenated:", np.concatenate(coeffs).shape)

# Which time sample does it use? Fill each channel with ch*1024 + t, then inspect
marked = (np.arange(18)[:, None] * 1024 + np.arange(1024)[None, :]).reshape(1, 18, 1024, 1)
used = marked[0, :, 0][:, 0]
print("time index used (first 3 channels):", (used - np.arange(18) * 1024)[:3])