"""
main.py
78차원 피처 1D-CNN 실시간 ASL 인식 및 Ollama 연동 스크립트
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from utils.text_manager import TextManager
from utils.ollama_manager import OllamaManager

MODEL_PATH = "assets/models/asl_1dcnn.pth"
CLASSES_PATH = "assets/models/classes.json"


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
            nn.MaxPool1d(2)
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
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    
    angles = []
    for p1, p2, p3 in joint_triplets:
        v1 = norm_pts[p1] - norm_pts[p2]
        v2 = norm_pts[p3] - norm_pts[p2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(np.arccos(cos_angle))

    return np.hstack([norm_pts.flatten(), np.array(angles)])


def main():
    print("[INFO] Initializing Inference Pipeline...")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print(f"[ERROR] Model file missing: {MODEL_PATH} or {CLASSES_PATH}")
        return

    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    model = AdvancedHandNet(in_features=78, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    text_manager = TextManager(threshold_count=7, cooldown_frames=10)
    # 속도 최우선 모델 qwen3.5:0.8b 적용 (필요 시 gemma3:4b로 변경 가능)
    ollama_mgr = OllamaManager(model_name="qwen2.5:3b")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("[FATAL] No video camera found!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[SUCCESS] Pipeline Ready.")
    print(" - Press 'c' to Clear Text")
    print(" - Press 'ENTER' or 'g' to Send Text to Ollama")
    print(" - Press 'q' or ESC to Exit")

    CONF_THRESHOLD = 0.70

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_pred = ""
        current_conf = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = extract_78d_features(hand_landmarks.landmark)
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_class = torch.max(probs, dim=1)

                current_pred = classes[pred_class.item()]
                current_conf = conf.item()

                if current_conf >= CONF_THRESHOLD:
                    text_manager.update(current_pred, current_conf)

        # 1. 상단: 현재 감지된 실시간 수어 정보
        if current_pred:
            label_text = f"Current: {current_pred} ({current_conf*100:.1f}%)"
            cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 2. 하단: 누적 텍스트 및 Ollama 응답 UI
        assembled_text = text_manager.get_text()
        ollama_result = ollama_mgr.get_response()

        # 하단 검은색 배경 오버레이 (100px 높이)
        cv2.rectangle(frame, (0, 380), (640, 480), (20, 20, 20), -1)

        cv2.putText(frame, f"Text  : {assembled_text}", (15, 415), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Ollama: {ollama_result}", (15, 455), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Jetson Orin Nano - ASL to Ollama", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            text_manager.clear()
            ollama_mgr.clear()
        elif key in (13, ord('g')):  # ENTER(13) 또는 'g' 키로 Ollama 호출
            prompt = text_manager.get_text()
            if prompt:
                print(f"[INFO] Sending to Ollama: {prompt}")
                system_prompt = (
                    "You are a helpful assistant for sign language interpretation. "
                    "Convert the given raw fingerspelled characters or words into a complete, "
                    "natural English sentence. Correct any spelling errors. "
                    "Output ONLY the corrected single sentence."
                )
                ollama_mgr.generate_async(prompt_text=prompt, system_prompt=system_prompt)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()