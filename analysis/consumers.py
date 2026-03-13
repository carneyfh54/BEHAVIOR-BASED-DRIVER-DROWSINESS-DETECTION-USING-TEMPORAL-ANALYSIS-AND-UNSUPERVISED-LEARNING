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

# ── Feature extraction ─────────────────────────────────────────────────────
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
        return None, []
    lm = results.multi_face_landmarks[0].landmark
    ear_left  = eye_aspect_ratio(lm, LEFT_EYE)
    ear_right = eye_aspect_ratio(lm, RIGHT_EYE)
    avg_ear   = (ear_left + ear_right) / 2.0
    mar       = mouth_aspect_ratio(lm)
    head_pitch = lm[152].y - lm[1].y
    features = np.array([ear_left, ear_right, avg_ear, mar, head_pitch], dtype=np.float32)
    landmarks = [{"x": l.x, "y": l.y} for l in lm]
    return features, landmarks

# ── Model architecture ─────────────────────────────────────────────────────
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

# Global fallback thresholds
GLOBAL_THRESHOLD_AWAKE    = thresholds['alert_mean'] + thresholds['alert_std']
GLOBAL_THRESHOLD_MILD     = thresholds['p95']
GLOBAL_THRESHOLD_MODERATE = thresholds['p99']

print(f"[WS] v2 model loaded | thresholds: mild={GLOBAL_THRESHOLD_MILD:.3f} moderate={GLOBAL_THRESHOLD_MODERATE:.3f}")

CALIBRATION_FRAMES = 150

# ── WebSocket consumer ─────────────────────────────────────────────────────
class VideoAnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.frame_counter   = 0
        self.feature_buffer  = []
        self.last_valid_feat = None
        self._last_landmarks = []

        # Calibration state
        self.calibrated         = False
        self.calibration_buffer = []
        self.threshold_awake    = GLOBAL_THRESHOLD_AWAKE
        self.threshold_mild     = GLOBAL_THRESHOLD_MILD
        self.threshold_moderate = GLOBAL_THRESHOLD_MODERATE
        self.error_history      = []
        self.drowsy_streak      = 0

        print("[WS] Client connected")

    async def disconnect(self, close_code):
        print(f"[WS] Client disconnected (code {close_code})")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                msg = json.loads(text_data)
                if msg.get("type") == "configure":
                    await self.send(text_data=json.dumps({"type": "configured"}))
            except Exception as e:
                print(f"[WS] Text message error: {e}")

        elif bytes_data:
            self.frame_counter += 1

            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            feat, landmarks = extract_features(frame)
            if feat is not None:
                self.last_valid_feat = feat
                self._last_landmarks = landmarks
            elif self.last_valid_feat is not None:
                feat = self.last_valid_feat
            else:
                return

            # ── CALIBRATION PHASE ──────────────────────────────────────────
            if not self.calibrated:
                self.calibration_buffer.append(feat)
                progress = len(self.calibration_buffer)

                await self.send(text_data=json.dumps({
                    "type":            "calibrating",
                    "frames_collected": progress,
                    "frames_needed":   CALIBRATION_FRAMES
                }))

                if progress >= CALIBRATION_FRAMES:
                    # Compute personal thresholds from calibration window
                    cal_errors = []
                    for i in range(len(self.calibration_buffer) - 89):
                        window = np.array(self.calibration_buffer[i:i+90], dtype=np.float32)
                        normalized = scaler.transform(window)
                        tensor = torch.from_numpy(normalized).unsqueeze(0)
                        with torch.no_grad():
                            recon = model(tensor)
                            error = torch.mean((recon - tensor) ** 2).item()
                        cal_errors.append(error)

                    cal_mean = float(np.mean(cal_errors))
                    cal_std  = float(np.std(cal_errors))

                    self.threshold_awake    = cal_mean + 2 * cal_std
                    self.threshold_mild     = cal_mean + 3 * cal_std
                    self.threshold_moderate = cal_mean + 4 * cal_std
                    self.calibrated         = True

                    # Seed feature buffer with calibration data
                    self.feature_buffer = list(self.calibration_buffer)

                    print(f"[CAL] Personal thresholds → awake={self.threshold_awake:.4f} mild={self.threshold_mild:.4f} moderate={self.threshold_moderate:.4f}")

                    await self.send(text_data=json.dumps({
                        "type":               "calibration_complete",
                        "threshold_awake":    round(self.threshold_awake, 4),
                        "threshold_mild":     round(self.threshold_mild, 4),
                        "threshold_moderate": round(self.threshold_moderate, 4)
                    }))
                return

            # ── DETECTION PHASE ────────────────────────────────────────────
            self.feature_buffer.append(feat)
            if len(self.feature_buffer) > 90:
                self.feature_buffer = self.feature_buffer[-90:]

            if len(self.feature_buffer) < 90:
                await self.send(text_data=json.dumps({
                    "type":             "buffering",
                    "frames_collected": len(self.feature_buffer),
                    "frames_needed":    90
                }))
                return

            window     = np.array(self.feature_buffer, dtype=np.float32)
            normalized = scaler.transform(window)
            tensor     = torch.from_numpy(normalized).unsqueeze(0)

            with torch.no_grad():
                recon = model(tensor)
                raw_error = torch.mean((recon - tensor) ** 2).item()

            # Smooth error over last 5 frames
            self.error_history.append(raw_error)
            if len(self.error_history) > 5:
                self.error_history = self.error_history[-5:]
            error = float(np.mean(self.error_history))

            # Classify raw level
            if error > self.threshold_moderate:
                raw_level = "highly drowsy"
            elif error > self.threshold_mild:
                raw_level = "moderately drowsy"
            elif error > self.threshold_awake:
                raw_level = "mildly drowsy"
            else:
                raw_level = "awake"

            # Consecutive frame confirmation
            if raw_level != "awake":
                self.drowsy_streak += 1
            else:
                self.drowsy_streak = 0

            # Only escalate after 3 consecutive drowsy frames
            if self.drowsy_streak >= 3:
                level = raw_level
            else:
                level = "awake"

            if level == "highly drowsy":
                conf   = min(0.99, 0.75 + (error - self.threshold_moderate) * 0.5)
                action = "PULL OVER AND REST IMMEDIATELY"
            elif level == "moderately drowsy":
                conf   = 0.75
                action = "Take a break soon"
            elif level == "mildly drowsy":
                conf   = 0.60
                action = "Stay alert"
            else:
                conf   = min(0.99, 0.90 - error)
                action = "Continue safely"

            cur = self.feature_buffer[-1]
            result = {
                "drowsiness_level":   level,
                "confidence":         round(conf, 2),
                "observations": [
                    f"Reconstruction error: {error:.4f}",
                    f"Personal baseline: {self.threshold_awake:.4f}",
                    f"EAR left: {cur[0]:.3f}  right: {cur[1]:.3f}",
                    f"MAR (mouth): {cur[3]:.3f}",
                    f"Head pitch: {cur[4]:.3f}"
                ],
                "recommended_action": action,
                "error_value":        round(error, 4)
            }

            print(f"[ML] Frame {self.frame_counter} → error={error:.4f} → {level} (baseline={self.threshold_awake:.4f})")

            await self.send(text_data=json.dumps({
                "type":         "analysis_result",
                "data":         result,
                "frame_number": self.frame_counter,
                "landmarks":    self._last_landmarks
            }))
