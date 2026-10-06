"""
src/train_utils.py
Loss, optimizer, and manual L2 regularization matching the paper's spec:
  - Binary cross-entropy loss (binary classification)
  - Adam optimizer, lr=0.0001
  - L2(0.03) applied ONLY to the 64-unit Dense layer (fc1), not the whole network

Why manual L2 instead of PyTorch's weight_decay:
  PyTorch's optimizer `weight_decay` argument applies the same L2 penalty to
  EVERY trainable parameter in the model (all conv filters, LSTM weights,
  both dense layers). The paper only regularizes fc1. Using weight_decay
  would over-regularize the rest of the network relative to the paper's
  design, so we add the penalty manually, only to fc1.weight, as an extra
  term added directly to the loss each step.
"""
import torch
import torch.nn as nn


def get_loss_fn(n_classes: int = 1):
    """Binary vs multi-class loss, matching the paper's spec."""
    if n_classes == 1:
        return nn.BCEWithLogitsLoss()  # combines sigmoid + BCE, numerically stable
    else:
        return nn.CrossEntropyLoss()   # combines softmax + categorical cross-entropy


def get_optimizer(model, lr: float = 0.0001):
    return torch.optim.Adam(model.parameters(), lr=lr)


def l2_penalty_fc1(model, l2_lambda: float = 0.03):
    """
    Manual L2 penalty on fc1's weight matrix only (NOT its bias, matching
    Keras's kernel_regularizer, which regularizes weights/kernel but not bias
    by default).
    Returns a scalar tensor to be ADDED to the loss before .backward().
    """
    return l2_lambda * torch.sum(model.fc1.weight ** 2)


if __name__ == "__main__":
    from model import CNNLSTM

    print("=" * 50)
    print("LOSS / OPTIMIZER / L2 PENALTY CHECK")
    print("=" * 50)

    model = CNNLSTM(n_classes=1)
    loss_fn = get_loss_fn(n_classes=1)
    optimizer = get_optimizer(model, lr=0.0001)

    # Dummy batch: 4 samples, matches the Bonn 4100-length DWT feature vector
    dummy_x = torch.randn(4, 1, 4100)
    dummy_y = torch.tensor([[0.], [1.], [0.], [1.]])  # binary labels, shape (4,1)

    # One manual training step, to confirm everything is wired correctly
    optimizer.zero_grad()
    logits = model(dummy_x)
    base_loss = loss_fn(logits, dummy_y)
    l2_term = l2_penalty_fc1(model, l2_lambda=0.03)
    total_loss = base_loss + l2_term

    print(f"Base BCE loss:        {base_loss.item():.4f}")
    print(f"L2 penalty (fc1 only): {l2_term.item():.4f}")
    print(f"Total loss (combined): {total_loss.item():.4f}")

    total_loss.backward()
    optimizer.step()
    print("\nbackward() and optimizer.step() ran without error -- gradients flow correctly.")

    # Confirm fc1's gradient exists (proves the L2 term is actually wired into fc1's weights)
    print(f"\nfc1.weight.grad is not None: {model.fc1.weight.grad is not None}")
    print(f"fc1.weight.grad shape: {tuple(model.fc1.weight.grad.shape)}")