"""
src/chbmit_quick_2x2.py
Quick PROXY experiment (Random Forest, NOT the paper's model) to see how much the
feature version and the split type matter on CHB-MIT, before spending GPU time.

Feature versions (both give 72 numbers per window):
  as_published : DWT of the FIRST time sample of each channel (what the authors' code does)
  band_energy  : per-channel z-score, DWT over the whole window,
                 log energy of each of the 4 sub-bands (A3, D3, D2, D1)
Splits:
  random  : KFold(10, shuffle=True, random_state=2) over windows (the paper's protocol)
  subject : GroupKFold(10) over subjects (chb21 merged into chb01: same person)
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold


def load_all(proc_dir):
    xs, ys, pats = [], [], []
    for f in sorted(glob.glob(os.path.join(proc_dir, "chb*.npz"))):
        d = np.load(f)
        xs.append(d["x"])
        ys.append(d["y"])
        pats.append(d["patient"])
    return np.concatenate(xs), np.concatenate(ys).astype(int), np.concatenate(pats)


def features_as_published(x):
    """Authors' code: DWT of time index 0 of each channel -> (18, 1) -> 72 numbers."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w in x:
            coeffs = pywt.wavedec(w[:, 0:1], "db1", level=3)
            out.append(np.concatenate(coeffs).ravel())
    return np.array(out)


def band_energy_features(x):
    x = x.astype(np.float32)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    coeffs = pywt.wavedec(x, "db1", level=3, axis=-1)          # [A3, D3, D2, D1]
    feats = [np.log((c ** 2).mean(axis=-1) + 1e-12) for c in coeffs]
    return np.concatenate(feats, axis=1)                        # (N, 72)


def evaluate(X, y, splits):
    accs, sens, specs, aucs = [], [], [], []
    for tr, te in splits:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        accs.append((tp + tn) / len(te))
        sens.append(tp / (tp + fn) if (tp + fn) else np.nan)
        specs.append(tn / (tn + fp) if (tn + fp) else np.nan)
        aucs.append(roc_auc_score(y[te], prob) if len(set(y[te])) > 1 else np.nan)
    return {k: (np.nanmean(v), np.nanstd(v))
            for k, v in dict(acc=accs, sens=sens, spec=specs, auc=aucs).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", required=True)
    args = ap.parse_args()

    x, y, pats = load_all(args.proc_dir)
    subjects = np.where(pats == "chb21", "chb01", pats)        # chb21 = same person as chb01
    print(f"windows: {len(y)} | seizure: {int(y.sum())} ({100 * y.mean():.1f}%) "
          f"| subjects: {len(set(subjects))}")
    print(f"majority-class accuracy (always 'non-seizure'): {100 * (1 - y.mean()):.1f}%\n")

    feats = {"as_published": features_as_published(x), "band_energy": band_energy_features(x)}
    for name, F in feats.items():
        print(f"features[{name}]: shape {F.shape}")

    splits = {
        "random": list(KFold(10, shuffle=True, random_state=2).split(x)),
        "subject": list(GroupKFold(n_splits=10).split(x, y, subjects)),
    }
    for sname, sp in splits.items():
        shared = [len(set(subjects[tr]) & set(subjects[te])) for tr, te in sp]
        print(f"split[{sname}]: subjects shared by train and test per fold: "
              f"mean {np.mean(shared):.1f} (max possible {len(set(subjects))})")
    print()

    print(f"{'features':14s} {'split':8s} {'accuracy':>16s} {'sensitivity':>16s} "
          f"{'specificity':>16s} {'AUC':>16s}")
    for fname, F in feats.items():
        for sname, sp in splits.items():
            r = evaluate(F, y, sp)
            cells = [f"{100 * r[k][0]:5.1f} +- {100 * r[k][1]:4.1f}" for k in ("acc", "sens", "spec", "auc")]
            print(f"{fname:14s} {sname:8s} " + " ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()