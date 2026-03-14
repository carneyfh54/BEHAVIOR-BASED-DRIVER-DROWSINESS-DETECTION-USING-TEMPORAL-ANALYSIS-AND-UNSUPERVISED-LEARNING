import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Load everything
ckpt = torch.load('lstm_autoencoder_v2.pth', map_location='cpu', weights_only=False)
train_losses = ckpt['train_losses']
val_losses   = ckpt['val_losses']

with open('scaler_v2.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('thresholds_v2.pkl', 'rb') as f:
    thresholds = pickle.load(f)

data = np.load('alert_sequences_v2.npy')
print(f"Data shape: {data.shape}")

THRESHOLD_AWAKE    = thresholds['alert_mean'] + thresholds['alert_std']
THRESHOLD_MILD     = thresholds['p95']
THRESHOLD_MODERATE = thresholds['p99']

# Plot 1: Training Loss Curve
fig, ax = plt.subplots(figsize=(10, 5))
epochs = range(1, len(train_losses) + 1)
ax.plot(epochs, train_losses, color='#4A90D9', linewidth=2, label='Training Loss')
ax.plot(epochs, val_losses,   color='#E74C3C', linewidth=2, label='Validation Loss')
ax.axhline(y=min(val_losses), color='#2ECC71', linestyle='--', linewidth=1.5,
           label=f'Best Val Loss: {min(val_losses):.4f}')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('MSE Loss', fontsize=13)
ax.set_title('LSTM Autoencoder Training & Validation Loss', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.tight_layout()
fig.savefig('plots/01_training_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 1 done")

# Plot 2: Dataset Comparison
fig, ax = plt.subplots(figsize=(8, 5))
versions = ['v1\n(YawDD basic)', 'v2\n(YawDD enhanced)']
counts   = [627, data.shape[0]]
colors   = ['#AED6F1', '#2E86C1']
bars = ax.bar(versions, counts, color=colors, width=0.4, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{count:,}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Sequences', fontsize=13)
ax.set_title('Dataset Size: v1 vs v2 Preprocessing', fontsize=15, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.tight_layout()
fig.savefig('plots/02_dataset_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 done")

# Plot 3: Feature Distributions
feature_names = ['EAR Left', 'EAR Right', 'Avg EAR', 'MAR', 'Head Pitch']
colors_feat   = ['#3498DB', '#2ECC71', '#9B59B6', '#E67E22', '#E74C3C']
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
sample = data[np.random.choice(data.shape[0], 5000, replace=False)]
flat   = sample.reshape(-1, 5)
for i, (ax, name, color) in enumerate(zip(axes, feature_names, colors_feat)):
    ax.hist(flat[:, i], bins=50, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8F9FA')
fig.suptitle('Feature Distributions Across Alert Sequences', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('plots/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 done")

# Plot 4: Sample Sequence
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
sample_seq = data[42]
colors_seq = ['#3498DB', '#2ECC71', '#9B59B6', '#E67E22', '#E74C3C']
for i, (ax, name, color) in enumerate(zip(axes, feature_names, colors_seq)):
    ax.plot(sample_seq[:, i], color=color, linewidth=1.8)
    ax.set_ylabel(name, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8F9FA')
axes[-1].set_xlabel('Frame', fontsize=11)
fig.suptitle('Sample Alert Sequence - 90 Frames of Facial Features', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('plots/04_sample_sequence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 4 done")

# Plot 5: Reconstruction Error Distribution
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(5, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = torch.nn.Linear(128, 32)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = torch.nn.Linear(32, 128)
        self.lstm = torch.nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.out  = torch.nn.Linear(128, 5)
    def forward(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, 90, 1)
        out, _ = self.lstm(x)
        return self.out(out)

class LSTMAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = LSTMAutoencoder()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("Computing reconstruction errors (this may take a minute)...")
sample_indices = np.random.choice(data.shape[0], 2000, replace=False)
errors = []
for idx, i in enumerate(sample_indices):
    seq = data[i]
    normalized = scaler.transform(seq)
    tensor = torch.from_numpy(normalized).unsqueeze(0).float()
    with torch.no_grad():
        recon = model(tensor)
        error = torch.mean((recon - tensor) ** 2).item()
    errors.append(error)
    if idx % 500 == 0:
        print(f"  {idx}/2000...")
errors = np.array(errors)

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(errors, bins=80, color='#3498DB', alpha=0.75, edgecolor='white', label='Alert sequences')
ax.axvline(THRESHOLD_AWAKE,    color='#F39C12', linewidth=2.5, linestyle='--', label=f'Awake: {THRESHOLD_AWAKE:.4f}')
ax.axvline(THRESHOLD_MILD,     color='#E67E22', linewidth=2.5, linestyle='--', label=f'Mild: {THRESHOLD_MILD:.4f}')
ax.axvline(THRESHOLD_MODERATE, color='#E74C3C', linewidth=2.5, linestyle='--', label=f'Moderate: {THRESHOLD_MODERATE:.4f}')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('Reconstruction Error Distribution with Drowsiness Thresholds', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.tight_layout()
fig.savefig('plots/05_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 5 done")

# Plot 6: Threshold Zones
fig, ax = plt.subplots(figsize=(10, 4))
x = np.linspace(0, 0.5, 1000)
ax.fill_between(x, 0, 1, where=(x <= THRESHOLD_AWAKE),                              color='#2ECC71', alpha=0.3, label='Awake')
ax.fill_between(x, 0, 1, where=(x > THRESHOLD_AWAKE) & (x <= THRESHOLD_MILD),       color='#F1C40F', alpha=0.3, label='Mildly Drowsy')
ax.fill_between(x, 0, 1, where=(x > THRESHOLD_MILD) & (x <= THRESHOLD_MODERATE),    color='#E67E22', alpha=0.3, label='Moderately Drowsy')
ax.fill_between(x, 0, 1, where=(x > THRESHOLD_MODERATE),                            color='#E74C3C', alpha=0.3, label='Highly Drowsy')
ax.axvline(THRESHOLD_AWAKE,    color='#27AE60', linewidth=2)
ax.axvline(THRESHOLD_MILD,     color='#D4AC0D', linewidth=2)
ax.axvline(THRESHOLD_MODERATE, color='#C0392B', linewidth=2)
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=13)
ax.set_title('Drowsiness Classification Threshold Zones', fontsize=15, fontweight='bold')
ax.set_yticks([])
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig('plots/06_threshold_zones.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 6 done")

print("\nAll plots saved to plots/ folder!")
