"""
src/baseline.py
Phase 13 sanity-check baseline — NOT the paper's model, NOT chasing
accuracy. Purpose: confirm dataset loading, labels, DWT features, and
evaluation all work correctly before building the real CNN-LSTM.
Task: Case 8 from the paper's Table 3 -- ABCD (non-seizure) vs E (seizure).
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    features = np.load("data/processed/bonn_features.npy")
    labels = np.load("data/processed/bonn_labels.npy")

    # Collapse to binary: label 4 (Set E, seizure) = 1, everything else (A,B,C,D) = 0
    binary_labels = (labels == 4).astype(int)

    print(f"Features shape: {features.shape}")
    print(f"Class balance -- seizure: {binary_labels.sum()}, "
          f"non-seizure: {len(binary_labels) - binary_labels.sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        features, binary_labels, test_size=0.2, stratify=binary_labels, random_state=42
    )
    print(f"Train/test sizes: {len(X_train)} / {len(X_test)}")

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["non-seizure", "seizure"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n" + "=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["non-seizure", "seizure"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))


if __name__ == "__main__":
    main()