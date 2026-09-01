"""
utils/asl_classifier.py
78차원 피처 추출, 1D-CNN (AdvancedHandNet) 추론 및 MediaPipe 연동 모듈
"""
import json
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


class AdvancedHandNet(nn.Module):

  def __init__(self, in_features, num_classes):
    super(AdvancedHandNet, self).__init__()
    self.conv_block = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.SiLU(),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.SiLU(),
        nn.MaxPool1d(2),
    )
    conv_out_dim = 128 * (in_features // 2)
    self.fc1 = nn.Linear(conv_out_dim, 256)
    self.bn1 = nn.BatchNorm1d(256)
    self.act1 = nn.SiLU()
    self.drop1 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(256, 128)
    self.bn2 = nn.BatchNorm1d(128)
    self.act2 = nn.SiLU()
    self.drop2 = nn.Dropout(0.3)
    self.out = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.conv_block(x)
    x = x.view(x.size(0), -1)
    x = self.drop1(self.act1(self.bn1(self.fc1(x))))
    x = self.drop2(self.act2(self.bn2(self.fc2(x))))
    return self.out(x)


def extract_78d_features(landmarks_21):
  pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_21])
  rel_pts = pts - pts[0]
  max_val = np.max(np.abs(rel_pts))
  norm_pts = rel_pts / max_val if max_val > 0 else rel_pts

  joint_triplets = [
      (0, 1, 2),
      (1, 2, 3),
      (2, 3, 4),
      (0, 5, 6),
      (5, 6, 7),
      (6, 7, 8),
      (0, 9, 10),
      (9, 10, 11),
      (10, 11, 12),
      (0, 13, 14),
      (13, 14, 15),
      (14, 15, 16),
      (0, 17, 18),
      (17, 18, 19),
      (18, 19, 20),
  ]

  angles = []
  for p1, p2, p3 in joint_triplets:
    v1 = norm_pts[p1] - norm_pts[p2]
    v2 = norm_pts[p3] - norm_pts[p2]
    cos_angle = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles.append(np.arccos(cos_angle))

  return np.hstack([norm_pts.flatten(), np.array(angles)])


class ASLClassifier:

  def __init__(self, model_path: str, classes_path: str):
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
      raise FileNotFoundError(
          f"[ERROR] Model file missing: {model_path} or {classes_path}"
      )

    with open(classes_path, "r") as f:
      self.classes = json.load(f)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {self.device}")

    self.model = AdvancedHandNet(
        in_features=78, num_classes=len(self.classes)
    ).to(self.device)
    self.model.load_state_dict(
        torch.load(model_path, map_location=self.device)
    )
    self.model.eval()

    self.mp_hands = mp.solutions.hands
    self.mp_draw = mp.solutions.drawing_utils
    self.hands = self.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

  def predict(self, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.hands.process(frame_rgb)

    current_pred = ""
    current_conf = 0.0

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )

        features = extract_78d_features(hand_landmarks.landmark)
        input_tensor = (
            torch.tensor(features, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
          outputs = self.model(input_tensor)
          probs = torch.softmax(outputs, dim=1)
          conf, pred_class = torch.max(probs, dim=1)

        current_pred = self.classes[pred_class.item()]
        current_conf = conf.item()

    return current_pred, current_conf

  def close(self):
    self.hands.close()