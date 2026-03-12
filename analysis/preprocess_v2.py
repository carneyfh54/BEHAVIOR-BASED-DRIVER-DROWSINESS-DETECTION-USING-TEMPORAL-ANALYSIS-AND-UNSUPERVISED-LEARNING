import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
import pickle
from sklearn.preprocessing import StandardScaler

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth: left corner, right corner, top-left, bottom-left, top-center, bottom-center
MOUTH     = [61, 291, 39, 181, 0, 17]

def eye_aspect_ratio(lm, indices):
    p = [np.array([lm[i].x, lm[i].y]) for i in indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def mouth_aspect_ratio(lm):
    p = [np.array([lm[i].x, lm[i].y]) for i in MOUTH]
    # vertical openings
    A = np.linalg.norm(p[2] - p[3])
    B = np.linalg.norm(p[4] - p[5])
    # horizontal width
    C = np.linalg.norm(p[0] - p[1])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def extract_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark

    ear_left  = eye_aspect_ratio(lm, LEFT_EYE)
    ear_right = eye_aspect_ratio(lm, RIGHT_EYE)
    avg_ear   = (ear_left + ear_right) / 2.0
    mar       = mouth_aspect_ratio(lm)

    # Head pitch proxy (same as your v1)
    head_pitch = lm[152].y - lm[1].y

    # 5 features total
    return np.array([ear_left, ear_right, avg_ear, mar, head_pitch], dtype=np.float32)

def process_video(video_path, seq_len=90, stride=1):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    last_valid = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feat = extract_features(frame)
        if feat is not None:
            last_valid = feat
            frame_features.append(feat)
        elif last_valid is not None:
            # Forward-fill instead of dropping — keeps sequence length consistent
            frame_features.append(last_valid)

    cap.release()

    if len(frame_features) < seq_len:
        return np.array([])

    # Sliding windows
    sequences = []
    for i in range(0, len(frame_features) - seq_len + 1, stride):
        sequences.append(frame_features[i:i + seq_len])
    return np.array(sequences)

# ── All 3 normal video folders ────────────────────────────────────────────────
NORMAL_DIRS = [
    "/Users/carney/Downloads/Yawdd/select_normal_mirror",
    "/Users/carney/Downloads/Yawdd/Mirror/Mirror/Female_mirror/Female_mirror_normal",
    "/Users/carney/Downloads/Yawdd/Mirror/Mirror/Male_mirror Avi Videos/Normal",
]

all_sequences = []
total_videos  = 0

for folder in NORMAL_DIRS:
    videos = [f for f in os.listdir(folder) if f.endswith((".avi", ".mp4"))]
    print(f"\nFolder: {folder}")
    print(f"  Videos found: {len(videos)}")
    for fname in tqdm(videos, desc="  Processing"):
        path = os.path.join(folder, fname)
        seqs = process_video(path)
        if len(seqs) > 0:
            all_sequences.append(seqs)
            total_videos += 1

print(f"\nTotal videos processed: {total_videos}")

# ── Concatenate + fit scaler ──────────────────────────────────────────────────
alert_sequences = np.concatenate(all_sequences, axis=0)
print(f"Total sequences (unscaled): {alert_sequences.shape}")

scaler = StandardScaler()
flat   = alert_sequences.reshape(-1, alert_sequences.shape[-1])
scaler.fit(flat)
normalized = scaler.transform(flat).reshape(alert_sequences.shape)

# ── Save ──────────────────────────────────────────────────────────────────────
np.save("alert_sequences_v2.npy", normalized)
with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\nDone.")
print(f"  Sequences shape : {normalized.shape}")
print(f"  Saved           : alert_sequences_v2.npy + scaler_v2.pkl")
print(f"  Features        : ear_left, ear_right, avg_ear, mar, head_pitch")