"""
src/model.py
1D CNN-LSTM model, reproducing the base paper's Table 2 architecture exactly,
for the Bonn dataset's 4100-length DWT feature vector input.

Keras -> PyTorch conversion notes:
  - BatchNorm momentum: Keras 0.9 == PyTorch 0.1 (inverted convention)
  - L2(0.03) on the 64-unit Dense layer only: NOT applied here via
    weight_decay (that would hit every parameter); applied manually in the
    training loop (Phase 15) to just this layer's weight tensor.
"""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()

        # --- Convolutional block (spatial feature extraction) ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(16, momentum=0.1)  # Keras momentum=0.9 -> PyTorch 0.1

        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(32, momentum=0.1)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(256, momentum=0.1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv6 = nn.Conv1d(256, 512, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm1d(512, momentum=0.1)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.relu = nn.ReLU()

        # --- Temporal feature extraction ---
        self.lstm = nn.LSTM(input_size=512, hidden_size=200, batch_first=True)

        # --- Classifier head ---
        self.fc1 = nn.Linear(200, 64)   # this is the layer that gets L2(0.03) in training
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, n_classes)

        self.debug = False  # set True to print shapes through the forward pass

    def _log(self, name, x):
        if self.debug:
            print(f"  after {name}: {tuple(x.shape)}")

    def forward(self, x):
        # x expected shape: (batch, 1, 4100)  -- channels-first, matching PyTorch Conv1d convention
        x = self.relu(self.bn1(self.conv1(x))); self._log("conv1", x)
        x = self.relu(self.bn2(self.conv2(x))); self._log("conv2", x)
        x = self.relu(self.bn3(self.conv3(x))); self._log("conv3", x)
        x = self.pool1(x); self._log("pool1", x)
        x = self.relu(self.bn4(self.conv4(x))); self._log("conv4", x)
        x = self.pool2(x); self._log("pool2", x)
        x = self.relu(self.bn5(self.conv5(x))); self._log("conv5", x)
        x = self.pool3(x); self._log("pool3", x)
        x = self.relu(self.bn6(self.conv6(x))); self._log("conv6", x)
        x = self.pool4(x); self._log("pool4", x)

        # LSTM expects (batch, seq_len, features) -- permute from (batch, channels, length)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)   # h_n: last hidden state, shape (1, batch, 200)
        x = h_n.squeeze(0)           # (batch, 200) -- equivalent to Keras LSTM(200) without return_sequences
        self._log("lstm (last hidden)", x)

        x = self.relu(self.fc1(x)); self._log("fc1", x)
        x = self.dropout(x)
        x = self.fc2(x)              # raw logits -- sigmoid applied in the loss function during training, not here
        self._log("fc2 (logits)", x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNLSTM(n_classes=1)
    model.debug = True

    print("=" * 50)
    print("MODEL ARCHITECTURE CHECK")
    print("=" * 50)
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    print(f"Paper's reported total (Bonn): 765,553")

    print("\n" + "=" * 50)
    print("DUMMY FORWARD PASS (batch of 4, length 4100)")
    print("=" * 50)
    dummy_input = torch.randn(4, 1, 4100)
    output = model(dummy_input)
    print(f"\nFinal output shape: {tuple(output.shape)}  (expected: (4, 1))")