"""
src/chbmit_extract.py
Extract CHB-MIT windows following the base paper's PUBLISHED code:
  - 18 bipolar channels (the authors' list; the paper text says 22)
  - 8 s windows, 4 s step, 256 Hz
  - a window is labelled seizure if ANY of its samples is inside a seizure
  - all seizure windows kept; 1% of non-seizure windows sampled per recording
  - microvolts, then decimated 2x (::2) -> 1024 samples per window
  - NO normalization (none appears in the authors' CHB-MIT code)
Every window keeps patient / file / start-second so patient-independent
splits (Experiment B) are possible later.
"""
import argparse
import glob
import os
import re

import numpy as np
import mne

mne.set_log_level("ERROR")

FS = 256
WIN = 8 * FS
STEP = 4 * FS
P_NONSEIZURE = 0.01
CH_LABELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ']


def parse_summary(summary_path):
    """Return {edf_filename: [(start_s, end_s), ...]} from a chbXX-summary.txt."""
    text = open(summary_path).read()
    seizures = {}
    for block in text.split("File Name:")[1:]:
        name = block.split()[0]
        s = [int(x) for x in re.findall(r"Seizure(?: \d+)? Start Time:\s*(\d+)", block)]
        e = [int(x) for x in re.findall(r"Seizure(?: \d+)? End Time:\s*(\d+)", block)]
        if s:
            seizures[name] = list(zip(s, e))
    return seizures


def pick_channels(ch_names):
    """Map each of the 18 labels to a channel in this file; None if any is missing.
    For duplicated names (T8-P8-0 / T8-P8-1) take the first, as the authors do."""
    chosen = []
    for c in CH_LABELS:
        matches = sorted(n for n in ch_names if n == c or n.startswith(c + "-"))
        if not matches:
            return None
        chosen.append(matches[0])
    return chosen


def extract_patient(raw_dir, patient, rng):
    pdir = os.path.join(raw_dir, patient)
    seizures = parse_summary(os.path.join(pdir, f"{patient}-summary.txt"))
    xs, ys, files, starts, skipped = [], [], [], [], []

    for edf_path in sorted(glob.glob(os.path.join(pdir, "*.edf"))):
        fname = os.path.basename(edf_path)
        raw = mne.io.read_raw_edf(edf_path, preload=False)
        picks = pick_channels(raw.ch_names)
        if picks is None or raw.info["sfreq"] != FS:
            skipped.append(fname)
            raw.close()
            continue
        data = (raw.get_data(picks=picks) * 1e6).astype(np.float32)   # (18, n), microvolts
        raw.close()
        n = data.shape[1]

        flag = np.zeros(n, dtype=bool)
        for s, e in seizures.get(fname, []):
            flag[s * FS:e * FS] = True
        n_win = (n - WIN) // STEP
        is_sz = np.array([flag[i * STEP:i * STEP + WIN].any() for i in range(n_win)])

        sz_idx = np.where(is_sz)[0]
        non_idx = np.where(~is_sz)[0]
        n_pick = round(P_NONSEIZURE * len(non_idx))
        non_pick = np.sort(rng.choice(non_idx, size=n_pick, replace=False)) if n_pick > 0 else []

        for idx, label in [(i, 1) for i in sz_idx] + [(i, 0) for i in non_pick]:
            w = data[:, idx * STEP: idx * STEP + WIN][:, ::2]   # 2048 -> 1024 samples
            xs.append(w)
            ys.append(label)
            files.append(fname)
            starts.append(idx * STEP // FS)                       # window start, seconds

        x = np.stack(xs).astype(np.float32)
        return dict(x=x, y=np.array(ys, dtype=np.int8), edf_file=np.array(files),
                start_s=np.array(starts, dtype=np.int32),
                patient=np.array([patient] * len(ys))), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=2023)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + int(args.patient[3:]))
    out, skipped = extract_patient(args.raw_dir, args.patient, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.patient}.npz")
    np.savez_compressed(out_path, **out)

    n_sz = int(out["y"].sum())
    print("=" * 50)
    print(f"{args.patient}: x shape = {out['x'].shape}")
    print(f"seizure windows: {n_sz} | non-seizure windows: {len(out['y']) - n_sz} "
          f"| seizure share: {100 * n_sz / len(out['y']):.1f}%")
    print(f"skipped files: {skipped}")
    print(f"saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()