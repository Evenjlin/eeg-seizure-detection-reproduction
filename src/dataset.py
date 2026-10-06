"""
src/dataset.py
Loads the Bonn EEG dataset from data/raw/bonn/{Z,O,N,F,S} and runs sanity checks.
"""
import numpy as np
from pathlib import Path

# Official mapping, verified against Andrzejak et al. (2001) and cross-checked
# against the base paper's own repo code (NOT the paper's prose, which has
# Set C / Set D swapped).
FOLDER_TO_CLASS = {
    "Z": {"set_name": "A", "label": 0, "description": "healthy, eyes open"},
    "O": {"set_name": "B", "label": 1, "description": "healthy, eyes closed"},
    "N": {"set_name": "C", "label": 2, "description": "interictal, hippocampus (opposite hemisphere)"},
    "F": {"set_name": "D", "label": 3, "description": "interictal, epileptogenic zone"},
    "S": {"set_name": "E", "label": 4, "description": "ictal (seizure)"},
}

EXPECTED_FILES_PER_FOLDER = 100


def load_bonn_dataset(raw_dir: str = "data/raw/bonn"):
    raw_dir = Path(raw_dir)
    all_signals = []
    all_labels = []
    all_set_names = []
    file_lengths = []

    for folder_letter, meta in FOLDER_TO_CLASS.items():
        folder_path = raw_dir / folder_letter
        if not folder_path.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder_path}")

        files = sorted(folder_path.glob("*.txt")) or sorted(folder_path.glob("*.TXT"))
        if len(files) != EXPECTED_FILES_PER_FOLDER:
            print(f"⚠️  WARNING: {folder_letter} has {len(files)} files, "
                  f"expected {EXPECTED_FILES_PER_FOLDER}")

        for f in files:
            signal = np.loadtxt(f)
            all_signals.append(signal)
            all_labels.append(meta["label"])
            all_set_names.append(meta["set_name"])
            file_lengths.append(len(signal))

    labels = np.array(all_labels)
    lengths = np.array(file_lengths)

    print("=" * 50)
    print("BONN DATASET LOAD SUMMARY")
    print("=" * 50)
    print(f"Total segments loaded: {len(all_signals)}")
    print(f"Expected total: {EXPECTED_FILES_PER_FOLDER * 5}")
    print(f"Unique signal lengths found: {sorted(set(lengths.tolist()))}")
    print(f"Min/Max length: {lengths.min()} / {lengths.max()}")
    print()
    print("Class distribution:")
    for letter, meta in FOLDER_TO_CLASS.items():
        count = np.sum(labels == meta["label"])
        print(f"  Set {meta['set_name']} ({letter}, label={meta['label']}, "
              f"{meta['description']}): {count} segments")
    print("=" * 50)

    return all_signals, labels, all_set_names


if __name__ == "__main__":
    signals, labels, set_names = load_bonn_dataset()