import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
import pickle
from sklearn.preprocessing import StandardScaler

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def extract_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    
    lm = results.multi_face_landmarks[0].landmark
    
    # Proper EAR (6-point per eye)
    def eye_aspect_ratio(eye_indices):
        p = [np.array([lm[i].x, lm[i].y]) for i in eye_indices]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C)
    
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]
    
    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    avg_ear = (ear_left + ear_right) / 2.0
    
    # ── Head pose approximations ───────────────────────────────────────────
    # Pitch: nose tip to chin (positive = looking down)
    nose_tip_y = lm[1].y
    chin_y = lm[152].y
    head_pitch = chin_y - nose_tip_y
    
    # Yaw proxy: left-right eye outer corners (positive = looking right)
    left_outer_x = lm[33].x
    right_outer_x = lm[263].x
    head_yaw = right_outer_x - left_outer_x
    
    # Return 3 features
    return np.array([avg_ear, head_pitch, head_yaw], dtype=np.float32)

def process_video(video_path, seq_len=90, stride=5):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            feat = extract_features(frame)
            if feat is not None:
                features.append(feat)

        frame_idx += 1

    cap.release()

    sequences = []
    for i in range(0, len(features) - seq_len + 1):
        seq = features[i:i + seq_len]
        sequences.append(seq)

    return np.array(sequences)

# ── CHANGE THIS PATH ─────────────────────────────────────────────────────────
# Point this to the folder containing your NORMAL/ALERT driving videos only
alert_dir = "/Users/carney/Downloads/Yawdd/select_normal_mirror"   # ← update this

all_sequences = []

for fname in tqdm(os.listdir(alert_dir)):
    if fname.endswith((".mp4", ".avi")):
        path = os.path.join(alert_dir, fname)
        seqs = process_video(path)
        if len(seqs) > 0:
            all_sequences.append(seqs)

if all_sequences:
    alert_sequences = np.concatenate(all_sequences, axis=0)

    # Normalize
    scaler = StandardScaler()
    flat = alert_sequences.reshape(-1, alert_sequences.shape[-1])
    scaler.fit(flat)
    normalized = scaler.transform(flat).reshape(alert_sequences.shape)

    np.save("alert_sequences.npy", normalized)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved {len(normalized)} sequences")
else:
    print("No sequences were generated. Check your video path and files.")
