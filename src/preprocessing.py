"""
src/preprocessing.py
Per-segment z-score normalization, matching the base paper's stated
methodology: "each window was normalized to have zero mean and unit variance."
"""
import numpy as np


def normalize_segment(signal: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalization of a single EEG segment.
    Applied per-segment (not using any dataset-wide statistic), so this
    is leakage-safe by construction -- no information from other segments
    (train or test) enters this calculation.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        std = 1.0  # guard against a constant/flat signal
    return (signal - mean) / std


def normalize_all(signals: list[np.ndarray]) -> np.ndarray:
    """Apply normalize_segment to a list of 1D signals, return as a 2D array."""
    normalized = np.array([normalize_segment(s) for s in signals])
    return normalized


if __name__ == "__main__":
    from dataset import load_bonn_dataset

    signals, labels, set_names = load_bonn_dataset()
    normalized = normalize_all(signals)

    print("=" * 50)
    print("NORMALIZATION CHECK")
    print("=" * 50)
    print(f"Shape: {normalized.shape}")
    print(f"Per-segment mean (should be ~0): {normalized.mean(axis=1)[:5]}")
    print(f"Per-segment std (should be ~1):  {normalized.std(axis=1)[:5]}")
    print(f"Global min/max after normalization: {normalized.min():.3f} / {normalized.max():.3f}")