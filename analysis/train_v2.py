import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle

# ── Config ────────────────────
DEVICE      = torch.device('cpu')
SEQ_LEN     = 90
N_FEATURES  = 5
BATCH_SIZE  = 64
EPOCHS      = 100
PATIENCE    = 15
LR          = 1e-3

print(f"Device: {DEVICE}")

# ── Load data ─────────────────────────────
data = np.load("alert_sequences_v2.npy").astype(np.float32)
print(f"Loaded: {data.shape}")

X_train, X_val = train_test_split(data, test_size=0.15, random_state=42, shuffle=True)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val)),   batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(128, 32)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(32, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.out  = nn.Linear(128, N_FEATURES)
    def forward(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        out, _ = self.lstm(x)
        return self.out(out)

class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

model     = LSTMAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
best_val_loss    = float('inf')
patience_count   = 0
best_state       = None

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    batch_losses = []
    for (xb,) in train_loader:
        xb = xb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), xb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)

    # Validate
    model.eval()
    with torch.no_grad():
        val_loss = np.mean([criterion(model(xb.to(DEVICE)), xb.to(DEVICE)).item() for (xb,) in val_loader])

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss - 1e-5:
        best_val_loss  = val_loss
        patience_count = 0
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1

    if epoch % 5 == 0 or patience_count == PATIENCE:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val_loss:.4f} | lr={lr_now:.2e} | patience={patience_count}/{PATIENCE}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# ── Save ──────────────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
torch.save({
    'model_state_dict': model.state_dict(),
    'best_val_loss':    best_val_loss,
    'train_losses':     train_losses,
    'val_losses':       val_losses,
    'n_features':       N_FEATURES,
    'seq_len':          SEQ_LEN,
}, 'lstm_autoencoder_v2.pth')

print(f"\nDone. Best val loss: {best_val_loss:.4f}")
print(f"Saved: lstm_autoencoder_v2.pth")
print(f"v1 was ~0.19-0.20 — improvement: {max(0, 0.195 - best_val_loss):.4f}")
