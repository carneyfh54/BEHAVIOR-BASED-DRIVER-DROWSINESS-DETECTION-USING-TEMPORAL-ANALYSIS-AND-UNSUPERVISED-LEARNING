import torch
import numpy as np
import pickle
import cv2
import mediapipe as mp
import os
import glob

with open('scaler_v2.pkl', 'rb') as f:
    scaler = pickle.load(f)

ckpt = torch.load('lstm_autoencoder_v2.pth', map_location='cpu', weights_only=False)

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
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH     = [61,291,39,181,0,17]

def ear(lm, idx):
    p = [np.array([lm[i].x, lm[i].y]) for i in idx]
    A = np.linalg.norm(p[1]-p[5]); B = np.linalg.norm(p[2]-p[4]); C = np.linalg.norm(p[0]-p[3])
    return (A+B)/(2*C) if C>0 else 0.0

def mar(lm):
    p = [np.array([lm[i].x, lm[i].y]) for i in MOUTH]
    A = np.linalg.norm(p[2]-p[3]); B = np.linalg.norm(p[4]-p[5]); C = np.linalg.norm(p[0]-p[1])
    return (A+B)/(2*C) if C>0 else 0.0

def extract(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = face_mesh.process(rgb)
    if not r.multi_face_landmarks: return None
    lm = r.multi_face_landmarks[0].landmark
    el = ear(lm, LEFT_EYE); er = ear(lm, RIGHT_EYE)
    return np.array([el, er, (el+er)/2, mar(lm), lm[152].y-lm[1].y], dtype=np.float32)

def process(path, max_seq=3):
    cap = cv2.VideoCapture(path)
    feats = []; last = None; seqs = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        f = extract(frame)
        if f is not None: last = f
        elif last is not None: f = last
        else: continue
        feats.append(f)
        if len(feats) >= 90:
            seq = np.array(feats[-90:], dtype=np.float32)
            norm = scaler.transform(seq)
            t = torch.from_numpy(norm).unsqueeze(0).float()
            with torch.no_grad():
                recon = model(t)
                e = torch.mean((recon-t)**2).item()
            seqs.append(e)
            if len(seqs) >= max_seq: break
    cap.release()
    return seqs

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

alert_errs = []
drowsy_errs = []

print("Processing normal...")
for i,v in enumerate(normal_videos[:20]):
    alert_errs.extend(process(v,2))
    if i%5==0: print(f"  {i+1}/20")

print("Processing drowsy...")
for i,v in enumerate(drowsy_videos[:20]):
    drowsy_errs.extend(process(v,2))
    if i%5==0: print(f"  {i+1}/20")

np.save('alert_errors.npy', np.array(alert_errs))
np.save('drowsy_errors.npy', np.array(drowsy_errs))
print(f"Saved! Alert: {len(alert_errs)}, Drowsy: {len(drowsy_errs)}")
