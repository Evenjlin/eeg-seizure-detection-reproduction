"""
src/chbmit_train.py
CNN-LSTM on CHB-MIT with a choice of FEATURE version and SPLIT type.

--features as_published : the authors' code: DWT of the first time sample of each channel (72 numbers,
                          54 of them exactly zero), no normalization
--features proper       : per-channel z-score, then db1 level-3 DWT over the whole 8 s window,
                          channels concatenated -> 18 x 1024 = 18432 numbers

--split random    : KFold(shuffle, seed 2) over windows            (the paper's protocol)
--split recording : GroupKFold over EDF recordings (no recording is in both train and test)
--split subject   : GroupKFold over people (chb21 merged into chb01)

Deviation from the paper (needed so the architecture runs on short inputs):
MaxPooling uses Keras-style padding='same' instead of 'valid'.
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

from chbmit_quick_2x2 import features_as_published
from model import CNNLSTM
from train_utils import get_optimizer, l2_penalty_fc1


# ----------------------------------------------------------------------------- model
class SamePadMaxPool1d(nn.Module):
    """Keras MaxPooling1D(pool_size=3, strides=2, padding='same')."""

    def __init__(self, k=3, s=2):
        super().__init__()
        self.k, self.s = k, s

    def forward(self, x):
        length = x.shape[-1]
        out = math.ceil(length / self.s)
        total = max((out - 1) * self.s + self.k - length, 0)
        left = total // 2
        x = F.pad(x, (left, total - left), value=float("-inf"))
        return F.max_pool1d(x, self.k, self.s)


class CNNLSTMSame(CNNLSTM):
    def __init__(self, n_classes=1):
        super().__init__(n_classes)
        self.pool1 = SamePadMaxPool1d()
        self.pool2 = SamePadMaxPool1d()
        self.pool3 = SamePadMaxPool1d()
        self.pool4 = SamePadMaxPool1d()


# ----------------------------------------------------------------------------- data
def load_all(proc_dir):
    xs, ys, pats, files = [], [], [], []
    for f in sorted(glob.glob(os.path.join(proc_dir, "chb*.npz"))):
        d = np.load(f)
        xs.append(d["x"])
        ys.append(d["y"])
        pats.append(d["patient"])
        files.append(d["edf_file"])
    x = np.concatenate(xs)
    y = np.concatenate(ys).astype(np.float32)
    patient = np.concatenate(pats).astype(str)
    edf = np.concatenate(files).astype(str)
    subject = np.where(patient == "chb21", "chb01", patient)      # chb21 = same person as chb01
    recording = np.char.add(np.char.add(patient, "/"), edf)
    return x, y, subject, recording


def features_proper(x):
    x = x.astype(np.float32)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    coeffs = pywt.wavedec(x, "db1", level=3, axis=-1)            # lengths 128, 128, 256, 512
    return np.concatenate(coeffs, axis=-1).reshape(len(x), -1).astype(np.float32)


def make_splits(kind, n, groups, n_folds, seed):
    idx = np.arange(n)
    if kind == "random":
        return list(KFold(n_folds, shuffle=True, random_state=seed).split(idx))
    return list(GroupKFold(n_splits=n_folds).split(idx, groups=groups))


def split_val(tr, kind, y, groups, seed, frac=0.15):
    """Hold out 15% of the training part for validation, using the same kind of split."""
    if kind == "random":
        return train_test_split(tr, test_size=frac, stratify=y[tr], random_state=seed)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed).split(tr, groups=groups[tr]))
    return tr[a], tr[b]


# ----------------------------------------------------------------------------- training
def evaluate(model, X, y, device, bs=512):
    model.eval()
    loss_sum, probs = 0.0, []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device)
            yb = torch.from_numpy(y[i:i + bs]).unsqueeze(1).to(device)
            logits = model(xb)
            loss_sum += F.binary_cross_entropy_with_logits(logits, yb, reduction="sum").item()
            probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return loss_sum / len(X), np.concatenate(probs)


def metrics(y, prob):
    yi = y.astype(int)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yi, pred, labels=[0, 1]).ravel()
    return dict(
        acc=(tp + tn) / len(yi),
        sens=tp / (tp + fn) if (tp + fn) else np.nan,
        spec=tn / (tn + fp) if (tn + fp) else np.nan,
        f1=f1_score(yi, pred, zero_division=0),
        mcc=matthews_corrcoef(yi, pred),
        auc=roc_auc_score(yi, prob) if len(set(yi.tolist())) > 1 else np.nan,
    )


def train_fold(Xtr, ytr, Xva, yva, Xte, yte, args, device, fold):
    model = CNNLSTMSame(n_classes=1).to(device)
    opt = get_optimizer(model, lr=args.lr)
    pos_weight = None
    if args.class_weight:
        ratio = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        pos_weight = torch.tensor([ratio], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state, bad, n = float("inf"), None, 0, len(Xtr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:                      # BatchNorm needs more than one sample
                continue
            xb = torch.from_numpy(Xtr[idx]).unsqueeze(1).to(device)
            yb = torch.from_numpy(ytr[idx]).unsqueeze(1).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb) + l2_penalty_fc1(model, args.l2)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        val_loss, _ = evaluate(model, Xva, yva, device)
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"    fold {fold} | epoch {epoch:3d} | train_loss {tot / max(nb, 1):.4f} | val_loss {val_loss:.4f}")
        if args.test_every and epoch % args.test_every == 0:
            _, p_diag = evaluate(model, Xte, yte, device)
            d = metrics(yte, p_diag)
            print(f"    fold {fold} | epoch {epoch:3d} | [diagnostic] test acc {d['acc']:.4f} "
                  f"sens {d['sens']:.4f} auc {d['auc']:.4f}")
        if args.patience and bad >= args.patience:
            print(f"    fold {fold} | early stop at epoch {epoch} (best val_loss {best_val:.4f})")
            break
    if best_state is not None and not args.eval_last:
        model.load_state_dict(best_state)
    _, prob = evaluate(model, Xte, yte, device)
    return metrics(yte, prob), epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--features", choices=["as_published", "proper"], required=True)
    ap.add_argument("--split", choices=["random", "recording", "subject"], required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="0 = no early stopping")
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=0.03)
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0, help="subsample for a quick smoke test")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--eval-last", action="store_true",
                    help="evaluate the LAST-epoch model (authors' protocol) instead of the best-validation checkpoint")
    ap.add_argument("--val-frac", type=float, default=0.15, help="paper: 0.15, authors' binary code: 0.2")
    ap.add_argument("--max-folds", type=int, default=0, help="stop after this many folds (0 = all)")
    ap.add_argument("--test-every", type=int, default=0,
                    help="DIAGNOSTIC ONLY: print test metrics every N epochs (never used for selection)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, subject, recording = load_all(args.proc_dir)
    if args.max_windows:
        sel = np.random.default_rng(args.seed).choice(len(y), args.max_windows, replace=False)
        x, y, subject, recording = x[sel], y[sel], subject[sel], recording[sel]
    print(f"device: {device} | windows: {len(y)} | seizure: {int(y.sum())} ({100 * y.mean():.1f}%) "
          f"| subjects: {len(set(subject))} | recordings: {len(set(recording))}")
    print(f"majority-class accuracy: {100 * (1 - y.mean()):.1f}%")

    X = (features_as_published(x) if args.features == "as_published" else features_proper(x)).astype(np.float32)
    del x
    print(f"features[{args.features}]: {X.shape}")

    groups = subject if args.split == "subject" else recording
    splits = make_splits(args.split, len(y), groups, args.folds, args.seed)

    rows, t0 = [], time.time()
    for fold, (tr, te) in enumerate(splits, 1):
        if args.max_folds and fold > args.max_folds:
            break
        tr_fit, tr_val = split_val(tr, args.split, y, groups, args.seed, args.val_frac)
        if args.split != "random":                # hard check: no group on both sides
            assert not (set(groups[tr_fit]) & set(groups[te])), "train/test group overlap!"
            assert not (set(groups[tr_val]) & set(groups[te])), "val/test group overlap!"
            assert not (set(groups[tr_fit]) & set(groups[tr_val])), "train/val group overlap!"
        print(f"fold {fold}/{len(splits)}: train {len(tr_fit)} | val {len(tr_val)} | test {len(te)} "
              f"| test subjects {len(set(subject[te]))}")
        m, ep = train_fold(X[tr_fit], y[tr_fit], X[tr_val], y[tr_val], X[te], y[te], args, device, fold)
        m.update(fold=fold, epochs_run=ep, n_test=len(te))
        rows.append(m)
        print(f"  fold {fold} TEST: acc {m['acc']:.4f} | sens {m['sens']:.4f} | spec {m['spec']:.4f} "
              f"| auc {m['auc']:.4f} | f1 {m['f1']:.4f}")

    df = pd.DataFrame(rows)
    minutes = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print(f"SUMMARY  features={args.features}  split={args.split}  folds={len(df)}  ({minutes:.1f} min)")
    for k in ["acc", "sens", "spec", "f1", "mcc", "auc"]:
        print(f"  {k:5s}: {df[k].mean():.4f} +- {df[k].std():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.features}_{args.split}" + ("_last" if args.eval_last else "") + ("_smoke" if args.max_windows else "")
    df.to_csv(os.path.join(args.out_dir, f"chbmit_{tag}.csv"), index=False)
    log = pd.DataFrame([dict(
        tag=tag, features=args.features, split=args.split, folds=len(df), epochs=args.epochs,
        patience=args.patience, eval_last=args.eval_last, val_frac=args.val_frac, l2=args.l2, class_weight=args.class_weight, windows=len(y),
        acc=df["acc"].mean(), sens=df["sens"].mean(), spec=df["spec"].mean(), auc=df["auc"].mean(),
        f1=df["f1"].mean(), minutes=round(minutes, 1))])
    log_path = os.path.join(args.out_dir, "chbmit_experiments_log.csv")
    log.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    print(f"saved: chbmit_{tag}.csv and appended a row to chbmit_experiments_log.csv")


if __name__ == "__main__":
    main()