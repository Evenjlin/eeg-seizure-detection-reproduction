"""
src/train.py
Phase 16: 10-fold CV training loop, reproducing the paper's protocol
(KFold(shuffle=True, random_state=2), 90/10 train-test per fold, 15%
of the 90% held out as validation), on the Bonn ABCD-vs-E binary task.

Run with SMOKE_TEST=True first (fast, CPU-friendly) to verify everything
works end-to-end. Switch to SMOKE_TEST=False for the real run on Colab.
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, cohen_kappa_score, roc_auc_score
)

from model import CNNLSTM
from train_utils import get_loss_fn, get_optimizer, l2_penalty_fc1

# ============================================================
# CONFIG -- flip this one flag to switch smoke test <-> full run
# ============================================================
SMOKE_TEST = True

if SMOKE_TEST:
    N_FOLDS = 3
    N_EPOCHS = 3
else:
    N_FOLDS = 10        # paper's spec
    N_EPOCHS = 300       # paper's spec

BATCH_SIZE = 60          # paper's spec, unchanged in both modes
LR = 0.0001
L2_LAMBDA = 0.03
VAL_SPLIT = 0.15
RANDOM_SEED = 2          # matches the paper's own KFold random_state


def set_all_seeds(seed: int):
    """Phase 18: reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, y_prob):
    """All metrics the paper reports, computed the same way their code does."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    gdr = np.sqrt(sensitivity * specificity)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "ppv": ppv,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "gdr": gdr,
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


def train_one_fold(X_train, y_train, X_val, y_val, X_test, y_test, fold_num, device):
    model = CNNLSTM(n_classes=1).to(device)
    loss_fn = get_loss_fn(n_classes=1)
    optimizer = get_optimizer(model, lr=LR)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (N, 1, 4100)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx].to(device), y_train_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb) + l2_penalty_fc1(model, L2_LAMBDA)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation pass (Phase 16: checkpoint selection uses VALIDATION, never test)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = loss_fn(val_logits, y_val_t.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, N_EPOCHS // 5) == 0 or epoch == 0:
            print(f"  Fold {fold_num} | Epoch {epoch+1}/{N_EPOCHS} | "
                  f"train_loss={epoch_loss/n_batches:.4f} | val_loss={val_loss:.4f}")

    # Restore best checkpoint (by validation loss) before evaluating on test
    model.load_state_dict(best_state)
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, f"outputs/checkpoints/bonn_fold{fold_num}_best.pt")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        test_prob = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_pred = (test_prob >= 0.5).astype(int)

    return compute_metrics(y_test, test_pred, test_prob)


def main():
    set_all_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'} "
          f"({N_FOLDS} folds x {N_EPOCHS} epochs)\n")

    features = np.load("data/processed/bonn_features.npy")
    labels = np.load("data/processed/bonn_labels.npy")
    binary_labels = (labels == 4).astype(int)  # ABCD vs E, same task as Phase 13 baseline

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold_num, (train_val_idx, test_idx) in enumerate(kf.split(features), start=1):
        print(f"\n{'='*50}\nFOLD {fold_num}/{N_FOLDS}\n{'='*50}")

        X_tr_va, X_test = features[train_val_idx], features[test_idx]
        y_tr_va, y_test = binary_labels[train_val_idx], binary_labels[test_idx]

        # 15% of the train+val portion held out as validation (paper's spec)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr_va, y_tr_va, test_size=VAL_SPLIT, stratify=y_tr_va, random_state=RANDOM_SEED
        )

        metrics = train_one_fold(X_train, y_train, X_val, y_val, X_test, y_test, fold_num, device)
        metrics["fold"] = fold_num
        fold_results.append(metrics)

        print(f"  Fold {fold_num} TEST results: "
              f"acc={metrics['accuracy']:.4f} | sens={metrics['sensitivity']:.4f} | "
              f"spec={metrics['specificity']:.4f} | f1={metrics['f1']:.4f}")

    results_df = pd.DataFrame(fold_results)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    tag = "smoketest" if SMOKE_TEST else "full"
    results_df.to_csv(f"outputs/metrics/bonn_experimentA_{tag}.csv", index=False)

    print(f"\n{'='*50}\nSUMMARY ACROSS {N_FOLDS} FOLDS ({tag.upper()})\n{'='*50}")
    for col in ["accuracy", "sensitivity", "specificity", "f1", "mcc", "kappa", "gdr", "auc"]:
        print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    print(f"\nPaper's reported Bonn accuracy (12-case average): 97.24% ± 0.38%")
    print(f"Saved per-fold results to outputs/metrics/bonn_experimentA_{tag}.csv")


if __name__ == "__main__":
    main()