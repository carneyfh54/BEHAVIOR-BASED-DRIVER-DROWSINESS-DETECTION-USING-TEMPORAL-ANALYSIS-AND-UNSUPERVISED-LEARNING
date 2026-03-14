import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('thresholds_v2.pkl', 'rb') as f:
    thresholds = pickle.load(f)

ckpt = torch.load('lstm_autoencoder_v2.pth', map_location='cpu', weights_only=False)
data = np.load('alert_sequences_v2.npy')

THRESHOLD_AWAKE    = thresholds['alert_mean'] + thresholds['alert_std']
THRESHOLD_MILD     = thresholds['p95']
THRESHOLD_MODERATE = thresholds['p99']

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

print("Computing reconstruction errors (no double scaling)...")
sample_indices = np.random.choice(data.shape[0], 2000, replace=False)
errors = []
for idx, i in enumerate(sample_indices):
    seq = data[i]  # already scaled
    tensor = torch.from_numpy(seq).unsqueeze(0).float()
    with torch.no_grad():
        recon = model(tensor)
        error = torch.mean((recon - tensor) ** 2).item()
    errors.append(error)
    if idx % 500 == 0:
        print(f"  {idx}/2000...")
errors = np.array(errors)

print(f"Error range: {errors.min():.4f} to {errors.max():.4f}")
print(f"Mean: {errors.mean():.4f}, Std: {errors.std():.4f}")
print(f"Thresholds: awake={THRESHOLD_AWAKE:.4f} mild={THRESHOLD_MILD:.4f} moderate={THRESHOLD_MODERATE:.4f}")

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
print("Plot 5 fixed!")
