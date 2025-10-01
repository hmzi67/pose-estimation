import cv2
import numpy as np
import torch
import os
import mediapipe as mp
from torch import nn
from utils import smooth_angles

os.environ['GLOG_minloglevel'] = '2'  # Suppress mediapipe INFO and WARNING logs

class HeadPoseMLP(nn.Module):
    def __init__(self, in_dim=1503, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class HeadPoseExtractor:
    def __init__(self, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HeadPoseMLP().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()
        self.x_mean = ckpt["x_mean"].to(self.device)
        self.x_std = ckpt["x_std"].to(self.device)
        self.y_mean = ckpt["y_mean"].to(self.device)
        self.y_std = ckpt["y_std"].to(self.device)

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.angle_buffer = []  # For smoothing

    def extract_features(self, frame):
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res_face = self.mp_face.process(img_rgb)
        if not res_face.multi_face_landmarks:
            return None
        face_lm = res_face.multi_face_landmarks[0].landmark
        face_pts = np.array([[p.x * w, p.y * h, p.z * w] for p in face_lm], dtype=np.float32)

        centroid = face_pts[:, :2].mean(axis=0)
        face_width = max(face_pts[:, 0].max() - face_pts[:, 0].min(), 1e-3)

        face_norm_xy = (face_pts[:, :2] - centroid) / face_width
        face_norm_z = face_pts[:, 2:3] / face_width
        face_flat = np.concatenate([face_norm_xy, face_norm_z], axis=1).flatten()

        res_pose = self.mp_pose.process(img_rgb)
        pose_flat = np.zeros(33 * 3, dtype=np.float32)
        if res_pose.pose_landmarks:
            pose_pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res_pose.pose_landmarks.landmark], dtype=np.float32)
            pose_norm_xy = (pose_pts[:, :2] - centroid) / face_width
            pose_norm_z = pose_pts[:, 2:3] / face_width
            pose_flat = np.concatenate([pose_norm_xy, pose_norm_z], axis=1).flatten()

        return np.concatenate([face_flat, pose_flat]).astype(np.float32)

    def process_frame(self, frame):
        feat = self.extract_features(frame)
        if feat is None:
            return None

        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        feat_norm = (feat_tensor - self.x_mean) / self.x_std

        with torch.no_grad():
            pred_norm = self.model(feat_norm)
            pred_deg = pred_norm * self.y_std + self.y_mean
            yaw_raw, pitch_raw, roll_raw = pred_deg[0].cpu().numpy()
            angles = np.array([-yaw_raw, -pitch_raw, roll_raw])  # Sign correction

        # Smooth angles
        self.angle_buffer.append(angles)
        if len(self.angle_buffer) > 5:  # Window size for smoothing
            self.angle_buffer.pop(0)
        if len(self.angle_buffer) >= 5:  # Ensure enough frames for smoothing
            smoothed_angles = smooth_angles(np.array(self.angle_buffer).T, window=5).T[-1]
            return smoothed_angles
        return angles  # Return unsmoothed if not enough frames

    def close(self):
        self.mp_face.close()
        self.mp_pose.close()