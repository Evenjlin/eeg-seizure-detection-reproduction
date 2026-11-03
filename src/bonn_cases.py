"""
src/bonn_cases.py
All 12 Bonn cases of the paper's Table 3, with the Bonn protocol used for the ABCD-vs-E run:
  10-fold KFold(shuffle, seed 2), 15% of the training part held out for validation,
  Adam lr 1e-4, batch 60, 300 epochs, L2(0.03) on fc1, best-validation-loss checkpoint.
Binary cases: 1 output + BCE. Multi-class cases: k outputs + cross-entropy (softmax).
Positive class for sensitivity/specificity = the second group (the epileptic / ictal side).
Each case is saved to its own CSV, so a disconnect loses at most one case.
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, train_test_split

from model import CNNLSTM
from train_utils import get_optimizer, l2_penalty_fc1

A, B, C, D, E = 0, 1, 2, 3, 4           # labels from dataset.py: Z=A, O=B, N=C, F=D, S=E
CASES = {
    1: ("A vs E", [[A], [E]]),
    2: ("B vs E", [[B], [E]]),
    3: ("AB vs E", [[A, B], [E]]),
    4: ("C vs E", [[C], [E]]),
    5: ("D vs E", [[D], [E]]),
    6: ("CD vs E", [[C, D], [E]]),
    7: ("A vs D", [[A], [D]]),
    8: ("ABCD vs E", [[A, B, C, D], [E]]),
    9: ("AB vs CDE", [[A, B], [C, D, E]]),
    10: ("A vs C vs E", [[A], [C], [E]]),
    11: ("AB vs CD vs E", [[A, B], [C, D], [E]]),
    12: ("A vs B vs C vs D vs E", [[A], [B], [C], [D], [E]]),
}


def make_case(X_all, labels, groups):
    y = np.full(len(labels), -1)
    for k, g in enumerate(groups):
        y[np.isin(labels, g)] = k
    keep = y >= 0
    return X_all[keep], y[keep]


def metrics(y, pred, k):
    out = dict(acc=float((pred == y).mean()),
               bal_acc=float(balanced_accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))
    if k == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out["sens"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["spec"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out


def run_fold(Xtr, ytr, Xva, yva, Xte, yte, k, args, device):
    model = CNNLSTM(n_classes=1 if k == 2 else k).to(device)
    opt = get_optimizer(model, lr=args.lr)

    def to_t(X):
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)

    Xtr_t, Xva_t, Xte_t = to_t(Xtr), to_t(Xva), to_t(Xte)
    ytr_t, yva_t = torch.tensor(ytr).to(device), torch.tensor(yva).to(device)

    def loss_fn(logits, yb):
        if k == 2:
            return F.binary_cross_entropy_with_logits(logits, yb.float().unsqueeze(1))
        return F.cross_entropy(logits, yb)

    best_val, best_state, n = float("inf"), None, len(Xtr)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:                      # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx]) + l2_penalty_fc1(model, args.l2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), yva_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {name: v.detach().clone() for name, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        if k == 2:
            pred = (torch.sigmoid(logits).cpu().numpy().ravel() >= 0.5).astype(int)
        else:
            pred = logits.argmax(1).cpu().numpy()
    return metrics(yte, pred, k)


def run_case(num, X_all, labels, args, device):
    name, groups = CASES[num]
    k = len(groups)
    X, y = make_case(X_all, labels, groups)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows, t0 = [], time.time()
    for fold, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=args.seed).split(X), 1):
        tr_fit, tr_val = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=args.seed)
        m = run_fold(X[tr_fit], y[tr_fit], X[tr_val], y[tr_val], X[te], y[te], k, args, device)
        m.update(case=num, name=name, fold=fold, n=len(y))
        rows.append(m)
        print(f"  case {num} ({name}) fold {fold}/{args.folds}: acc {m['acc']:.4f}")
    print(f"case {num} ({name}): {len(y)} windows, {k} classes, {(time.time() - t0) / 60:.1f} min")
    return pd.DataFrame(rows)


def summarize(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "bonn_case*.csv")))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    cols = [c for c in ["acc", "bal_acc", "macro_f1", "sens", "spec"] if c in df.columns]
    summ = df.groupby(["case", "name"])[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index().sort_values("case")
    print("\n" + "=" * 70)
    print(f"{'case':>4s}  {'name':22s} {'accuracy %':>16s} {'sens %':>8s} {'spec %':>8s} {'macro-F1 %':>11s}")
    for _, r in summ.iterrows():
        sens = f"{100 * r['sens_mean']:8.1f}" if "sens_mean" in r and not np.isnan(r["sens_mean"]) else "       -"
        spec = f"{100 * r['spec_mean']:8.1f}" if "spec_mean" in r and not np.isnan(r["spec_mean"]) else "       -"
        print(f"{int(r['case']):4d}  {r['name']:22s} {100 * r['acc_mean']:7.2f} +- {100 * r['acc_std']:4.2f} {sens} {spec} {100 * r['macro_f1_mean']:11.1f}")
    print(f"\ncases finished: {len(summ)} of 12")
    print(f"mean accuracy over finished cases: {100 * summ['acc_mean'].mean():.2f}%   (paper: 97.24% averaged over 12 cases)")
    summ.to_csv(os.path.join(out_dir, "bonn_cases_summary.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="re-run cases whose CSV already exists")
    ap.add_argument("--features", default="data/processed/bonn_features.npy")
    ap.add_argument("--labels", default="data/processed/bonn_labels.npy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_all, labels = np.load(args.features), np.load(args.labels)
    print(f"device: {device} | features {X_all.shape} | labels {np.bincount(labels).tolist()}")
    os.makedirs(args.out_dir, exist_ok=True)

    for num in [int(c) for c in args.cases.split(",")]:
        path = os.path.join(args.out_dir, f"bonn_case{num:02d}.csv")
        if os.path.exists(path) and not args.force:
            print(f"skip case {num} (already done: {path})")
            continue
        run_case(num, X_all, labels, args, device).to_csv(path, index=False)
    summarize(args.out_dir)


if __name__ == "__main__":
    main()