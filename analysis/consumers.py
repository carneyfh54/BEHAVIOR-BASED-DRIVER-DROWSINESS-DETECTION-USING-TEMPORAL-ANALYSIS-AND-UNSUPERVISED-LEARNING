import json
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import pickle
from channels.generic.websocket import AsyncWebsocketConsumer

# ── MediaPipe setup ────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Landmark indices ───────────────────────────────────────────────────────
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 39, 181, 0, 17]

# ── Feature extraction (v2 — 5 features) ──────────────────────────────────
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
    ear_left  = eye_aspect_ratio(lm, LEFT_EYE)
    ear_right = eye_aspect_ratio(lm, RIGHT_EYE)
    avg_ear   = (ear_left + ear_right) / 2.0
    mar       = mouth_aspect_ratio(lm)
    head_pitch = lm[152].y - lm[1].y
    return np.array([ear_left, ear_right, avg_ear, mar, head_pitch], dtype=np.float32)

# ── v2 Model architecture (must match training exactly) ───────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(128, 32)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(32, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.out  = nn.Linear(128, 5)
    def forward(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, 90, 1)
        out, _ = self.lstm(x)
        return self.out(out)

class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Load model, scaler, thresholds ────────────────────────────────────────
ckpt  = torch.load("lstm_autoencoder_v2.pth", map_location="cpu", weights_only=False)
model = LSTMAutoencoder()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with open("scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("thresholds_v2.pkl", "rb") as f:
    thresholds = pickle.load(f)

THRESHOLD_AWAKE    = thresholds['alert_mean'] + thresholds['alert_std']   # 0.1216
THRESHOLD_MILD     = thresholds['p95']                                     # 0.1597
THRESHOLD_MODERATE = thresholds['p99']                                     # 0.3509

print(f"[WS] v2 model loaded | thresholds: mild={THRESHOLD_MILD:.3f} moderate={THRESHOLD_MODERATE:.3f}")

# ── WebSocket consumer ─────────────────────────────────────────────────────
class VideoAnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.frame_counter  = 0
        self.feature_buffer = []
        self.last_valid_feat = None
        print("[WS] Client connected")

    async def disconnect(self, close_code):
        print(f"[WS] Client disconnected (code {close_code})")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                msg = json.loads(text_data)
                if msg.get("type") == "configure":
                    interval = msg.get("interval", 1)
                    await self.send(text_data=json.dumps({
                        "type": "configured",
                        "interval": interval
                    }))
            except Exception as e:
                print(f"[WS] Text message error: {e}")

        elif bytes_data:
            self.frame_counter += 1

            # Decode JPEG
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            # Extract 5 features
            feat = extract_features(frame)
            if feat is not None:
                self.last_valid_feat = feat
            elif self.last_valid_feat is not None:
                feat = self.last_valid_feat  # forward-fill if no face
            else:
                return  # no face yet, skip

            # Buffer management
            self.feature_buffer.append(feat)
            if len(self.feature_buffer) > 90:
                self.feature_buffer = self.feature_buffer[-90:]

            # Only run inference on full window
            if len(self.feature_buffer) < 90:
                await self.send(text_data=json.dumps({
                    "type": "buffering",
                    "frames_collected": len(self.feature_buffer),
                    "frames_needed": 90
                }))
                return

            # Normalize + inference
            window     = np.array(self.feature_buffer, dtype=np.float32)   # (90, 5)
            normalized = scaler.transform(window)                           # (90, 5)
            tensor     = torch.from_numpy(normalized).unsqueeze(0)         # (1, 90, 5)

            with torch.no_grad():
                recon = model(tensor)
                error = torch.mean((recon - tensor) ** 2).item()

            # Map error to drowsiness level
            if error > THRESHOLD_MODERATE:
                level  = "highly drowsy"
                conf   = min(0.99, 0.75 + (error - THRESHOLD_MODERATE) * 0.5)
                action = "PULL OVER AND REST IMMEDIATELY"
            elif error > THRESHOLD_MILD:
                level  = "moderately drowsy"
                conf   = 0.75
                action = "Take a break soon"
            elif error > THRESHOLD_AWAKE:
                level  = "mildly drowsy"
                conf   = 0.60
                action = "Stay alert"
            else:
                level  = "awake"
                conf   = min(0.99, 0.90 - error)
                action = "Continue safely"

            # Current frame features for display
            cur = self.feature_buffer[-1]
            result = {
                "drowsiness_level":   level,
                "confidence":         round(conf, 2),
                "observations": [
                    f"Reconstruction error: {error:.4f}",
                    f"EAR left: {cur[0]:.3f}  right: {cur[1]:.3f}",
                    f"MAR (mouth): {cur[3]:.3f}",
                    f"Head pitch: {cur[4]:.3f}"
                ],
                "recommended_action": action,
                "error_value":        round(error, 4)
            }

            print(f"[ML] Frame {self.frame_counter} → error={error:.4f} → {level}")

            await self.send(text_data=json.dumps({
                "type":         "analysis_result",
                "data":         result,
                "frame_number": self.frame_counter
            }))
