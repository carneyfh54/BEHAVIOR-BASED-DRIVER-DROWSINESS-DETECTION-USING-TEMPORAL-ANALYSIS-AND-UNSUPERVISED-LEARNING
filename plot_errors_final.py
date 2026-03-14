import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('thresholds_v2.pkl', 'rb') as f:
    thresholds = pickle.load(f)

THRESHOLD_AWAKE    = thresholds['alert_mean'] + thresholds['alert_std']
THRESHOLD_MILD     = thresholds['p95']
THRESHOLD_MODERATE = thresholds['p99']

alert_errs  = np.load('alert_errors.npy')
drowsy_errs = np.load('drowsy_errors.npy')

print(f"Alert  — mean: {alert_errs.mean():.4f}, range: {alert_errs.min():.4f}-{alert_errs.max():.4f}")
print(f"Drowsy — mean: {drowsy_errs.mean():.4f}, range: {drowsy_errs.min():.4f}-{drowsy_errs.max():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: zoomed to 0-1.0 to show threshold separation
ax1.hist(alert_errs,                        bins=20, color='#2ECC71', alpha=0.8, label='Alert',  edgecolor='white')
ax1.hist(drowsy_errs[drowsy_errs <= 1.0],   bins=20, color='#E74C3C', alpha=0.8, label='Drowsy (≤1.0)', edgecolor='white')
ax1.axvline(THRESHOLD_AWAKE,    color='#F39C12', linewidth=2.5, linestyle='--', label=f'Awake: {THRESHOLD_AWAKE:.4f}')
ax1.axvline(THRESHOLD_MILD,     color='#E67E22', linewidth=2.5, linestyle='--', label=f'Mild: {THRESHOLD_MILD:.4f}')
ax1.axvline(THRESHOLD_MODERATE, color='#C0392B', linewidth=2.5, linestyle='--', label=f'Moderate: {THRESHOLD_MODERATE:.4f}')
ax1.set_xlim(0, 1.0)
ax1.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Zoomed View (0 – 1.0)\nShowing Threshold Separation', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F8F9FA')

# Right panel: full range
ax2.hist(alert_errs,  bins=30, color='#2ECC71', alpha=0.8, label='Alert sequences',  edgecolor='white')
ax2.hist(drowsy_errs, bins=30, color='#E74C3C', alpha=0.8, label='Drowsy sequences', edgecolor='white')
ax2.axvline(THRESHOLD_AWAKE,    color='#F39C12', linewidth=2.5, linestyle='--', label=f'Awake: {THRESHOLD_AWAKE:.4f}')
ax2.axvline(THRESHOLD_MILD,     color='#E67E22', linewidth=2.5, linestyle='--', label=f'Mild: {THRESHOLD_MILD:.4f}')
ax2.axvline(THRESHOLD_MODERATE, color='#C0392B', linewidth=2.5, linestyle='--', label=f'Moderate: {THRESHOLD_MODERATE:.4f}')
ax2.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Full Range View\nShowing Complete Error Spread', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F8F9FA')

fig.suptitle('Reconstruction Error: Alert vs Drowsy Sequences', fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig('plots/08_alert_vs_drowsy_errors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 8 saved!")
