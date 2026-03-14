import torch
import numpy as np
import pickle
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import glob

with open('scaler_v2.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('thresholds_v2.pkl', 'rb') as f:
    thresholds = pickle.load(f)

ckpt = torch.load('lstm_autoencoder_v2.pth', map_location='cpu', weights_only=False)

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 39, 181, 0, 17]

def eye_aspect_ratio(lm, indices):
    p = [np.array([lm[i].x, lm[i].y]) for i in indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def mouth_aspect_ratio(lm):
    p = [np.array([lm[i].x, lm[i].y]) for i in MOUTH]
    A = np.linalg.norm(p[2] - p[3])
    B = np.linalg.norm(p[4] - p[5])
    C = np.linalg.norm(p[0] - p[1])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def extract_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    ear_left   = eye_aspect_ratio(lm, LEFT_EYE)
    ear_right  = eye_aspect_ratio(lm, RIGHT_EYE)
    avg_ear    = (ear_left + ear_right) / 2.0
    mar        = mouth_aspect_ratio(lm)
    head_pitch = lm[152].y - lm[1].y
    return np.array([ear_left, ear_right, avg_ear, mar, head_pitch], dtype=np.float32)

def process_video(video_path, max_sequences=3):
    cap = cv2.VideoCapture(video_path)
    features = []
    last_valid = None
    sequences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = extract_features(frame)
        if feat is not None:
            last_valid = feat
        elif last_valid is not None:
            feat = last_valid
        else:
            continue
        features.append(feat)

        if len(features) >= 90:
            seq = np.array(features[-90:], dtype=np.float32)
            normalized = scaler.transform(seq)  # APPLY SCALER to raw features
            tensor = torch.from_numpy(normalized).unsqueeze(0).float()
            with torch.no_grad():
                recon = model(tensor)
                error = torch.mean((recon - tensor) ** 2).item()
            sequences.append(error)
            if len(sequences) >= max_sequences:
                break

    cap.release()
    return sequences

def classify(error):
    if error > THRESHOLD_MODERATE:
        return 'highly drowsy'
    elif error > THRESHOLD_MILD:
        return 'moderately drowsy'
    elif error > THRESHOLD_AWAKE:
        return 'mildly drowsy'
    else:
        return 'awake'

base = os.path.expanduser('~/Downloads/Yawdd')

normal_videos = (
    glob.glob(f'{base}/Mirror/Mirror/Female_mirror/Female_mirror_normal/*.avi') +
    glob.glob(f'{base}/Mirror/Mirror/Male_mirror Avi Videos/Normal/*.avi') +
    glob.glob(f'{base}/Dash/Dash/Female/*.avi') +
    glob.glob(f'{base}/Dash/Dash/Male/*.avi')
)

drowsy_videos = (
    glob.glob(f'{base}/Mirror/Mirror/Female_mirror/Female_mirror_yawning/*.avi') +
    glob.glob(f'{base}/Mirror/Mirror/Male_mirror Avi Videos/Yawning/*.avi')
)

print(f"Found {len(normal_videos)} normal, {len(drowsy_videos)} drowsy videos")

true_labels = []
pred_labels = []
errors_alert  = []
errors_drowsy = []

print("\nProcessing normal videos...")
for i, vpath in enumerate(normal_videos[:20]):
    errs = process_video(vpath, max_sequences=2)
    for e in errs:
        true_labels.append('alert')
        pred = classify(e)
        pred_binary = 'alert' if pred in ['awake', 'mildly drowsy'] else 'drowsy'
        pred_labels.append(pred_binary)
        errors_alert.append(e)
    if i % 5 == 0:
        print(f"  {i+1}/{min(20, len(normal_videos))}...")

print("Processing drowsy videos...")
for i, vpath in enumerate(drowsy_videos[:20]):
    errs = process_video(vpath, max_sequences=2)
    for e in errs:
        true_labels.append('drowsy')
        pred = classify(e)
        pred_binary = 'alert' if pred in ['awake', 'mildly drowsy'] else 'drowsy'
        pred_labels.append(pred_binary)
        errors_drowsy.append(e)
    if i % 5 == 0:
        print(f"  {i+1}/{min(20, len(drowsy_videos))}...")

print(f"\nTotal: {len(true_labels)} (alert: {true_labels.count('alert')}, drowsy: {true_labels.count('drowsy')})")

cm = confusion_matrix(true_labels, pred_labels, labels=['alert', 'drowsy'])
report = classification_report(true_labels, pred_labels, labels=['alert', 'drowsy'], zero_division=0)
print("\nClassification Report:")
print(report)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Alert', 'Predicted Drowsy'], fontsize=12)
ax.set_yticklabels(['Actual Alert', 'Actual Drowsy'], fontsize=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=18, fontweight='bold',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
ax.set_title('Confusion Matrix — Alert vs Drowsy Detection', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
fig.tight_layout()
fig.savefig('plots/07_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved!")

# Plot alert vs drowsy errors
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(errors_alert,  bins=40, color='#2ECC71', alpha=0.6, label='Alert sequences',  edgecolor='white')
ax.hist(errors_drowsy, bins=40, color='#E74C3C', alpha=0.6, label='Drowsy sequences', edgecolor='white')
ax.axvline(THRESHOLD_AWAKE,    color='#F39C12', linewidth=2, linestyle='--', label=f'Awake: {THRESHOLD_AWAKE:.4f}')
ax.axvline(THRESHOLD_MILD,     color='#E67E22', linewidth=2, linestyle='--', label=f'Mild: {THRESHOLD_MILD:.4f}')
ax.axvline(THRESHOLD_MODERATE, color='#C0392B', linewidth=2, linestyle='--', label=f'Moderate: {THRESHOLD_MODERATE:.4f}')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('Reconstruction Error: Alert vs Drowsy Sequences', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.tight_layout()
fig.savefig('plots/08_alert_vs_drowsy_errors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Alert vs drowsy plot saved!")
