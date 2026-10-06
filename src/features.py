"""
src/features.py
DWT feature extraction, matching the base paper's own GitHub code exactly:
    pywt.wavedec(signal, 'db1', level=3)
then concatenating all resulting coefficient arrays (A3, D3, D2, D1) into
one 1D feature vector.
"""
import numpy as np
import pywt


def extract_dwt_features(signal: np.ndarray, wavelet: str = "db1", level: int = 3) -> np.ndarray:
    """
    Decompose a single 1D EEG signal with DWT and concatenate all
    coefficient bands into one 1D feature vector.
    coeffs returned by pywt.wavedec = [A3, D3, D2, D1] (approx first, then
    details from coarsest to finest).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)


def extract_dwt_features_batch(signals: np.ndarray, wavelet: str = "db1", level: int = 3) -> np.ndarray:
    """Apply extract_dwt_features to every row in a 2D array of signals."""
    features = [extract_dwt_features(s, wavelet, level) for s in signals]
    return np.array(features)


if __name__ == "__main__":
    from dataset import load_bonn_dataset
    from preprocessing import normalize_all

    signals, labels, set_names = load_bonn_dataset()
    normalized = normalize_all(signals)

    features = extract_dwt_features_batch(normalized, wavelet="db1", level=3)

    print("=" * 50)
    print("DWT FEATURE EXTRACTION CHECK")
    print("=" * 50)
    print(f"Input signal shape:  {normalized.shape}")
    print(f"Output feature shape: {features.shape}")
    print(f"Expected feature length per the paper: 4100")
    print(f"Actual feature length: {features.shape[1]}")

    # Save processed features + labels for the next step (model building)
    np.save("data/processed/bonn_features.npy", features)
    np.save("data/processed/bonn_labels.npy", labels)
    print("\nSaved: data/processed/bonn_features.npy, data/processed/bonn_labels.npy")