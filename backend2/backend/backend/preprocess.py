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

    # Simple features (expand later: EAR left/right, yaw/pitch/roll, blink proxy)
    # Example: average vertical eye opening + head tilt
    left_eye_y = [lm[i].y for i in [159, 145]]  # upper/lower lid approx
    right_eye_y = [lm[i].y for i in [386, 374]]
    ear_approx = (np.mean(left_eye_y) - np.mean(right_eye_y))  # simplistic

    nose_y = lm[1].y
    chin_y = lm[152].y
    head_tilt = chin_y - nose_y  # rough downward tilt proxy

    return np.array([ear_approx, head_tilt])  # Add more!

def process_video(video_path, seq_len=90, stride=5):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

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

# Process alert videos only
alert_dir = "path/to/uta-rldd/alert"  # change to your folder
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