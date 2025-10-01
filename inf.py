import cv2
import numpy as np
import torch
import mediapipe as mp
from torch import nn

# ----------------------------
# 1. Load your trained model
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the same MLP architecture
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

# Load checkpoint (adjust path if needed)
ckpt = torch.load("model/state_of_the_art.pth", map_location=DEVICE)
model = HeadPoseMLP().to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load normalization stats
x_mean = ckpt["x_mean"].to(DEVICE)
x_std = ckpt["x_std"].to(DEVICE)
y_mean = ckpt["y_mean"].to(DEVICE)
y_std = ckpt["y_std"].to(DEVICE)

print("✅ Model and stats loaded.")

# ----------------------------
# 2. Initialize MediaPipe
# ----------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# 3. Feature extraction (MUST match training!)
# ----------------------------
def extract_features_live(frame):
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face
    res_face = mp_face.process(img_rgb)
    if not res_face.multi_face_landmarks:
        return None
    face_lm = res_face.multi_face_landmarks[0].landmark
    face_pts = np.array([[p.x * w, p.y * h, p.z * w] for p in face_lm], dtype=np.float32)

    centroid = face_pts[:, :2].mean(axis=0)
    face_width = max(face_pts[:, 0].max() - face_pts[:, 0].min(), 1e-3)

    face_norm_xy = (face_pts[:, :2] - centroid) / face_width
    face_norm_z = face_pts[:, 2:3] / face_width
    face_flat = np.concatenate([face_norm_xy, face_norm_z], axis=1).flatten()

    # Pose
    res_pose = mp_pose.process(img_rgb)
    pose_flat = np.zeros(33 * 3, dtype=np.float32)
    if res_pose.pose_landmarks:
        pose_pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res_pose.pose_landmarks.landmark], dtype=np.float32)
        pose_norm_xy = (pose_pts[:, :2] - centroid) / face_width
        pose_norm_z = pose_pts[:, 2:3] / face_width
        pose_flat = np.concatenate([pose_norm_xy, pose_norm_z], axis=1).flatten()

    feat = np.concatenate([face_flat, pose_flat]).astype(np.float32)
    return feat

# ----------------------------
# 4. Live Inference Loop
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Extract features
    feat = extract_features_live(frame)
    if feat is not None:
        # Normalize input
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        feat_norm = (feat_tensor - x_mean) / x_std

        # Predict
        with torch.no_grad():
            pred_norm = model(feat_norm)
            pred_deg = pred_norm * y_std + y_mean
            yaw_raw, pitch_raw, roll_raw = pred_deg[0].cpu().numpy()
            # Apply sign correction to match intuitive directions:
            yaw   = -yaw_raw    # Right turn → positive
            pitch = -pitch_raw  # Look up → positive
            roll  = roll_raw    # Roll usually matches (left ear down = positive)

            print(f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°, Roll: {roll:.1f}°")

        # Display on frame
        cv2.putText(frame, f"Yaw:   {yaw:6.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:6.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll:  {roll:6.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Yaw:   {yaw:6.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {pitch:6.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Roll:  {roll:6.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Live Head Pose Estimation (BIWI MLP)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mp_face.close()
mp_pose.close()