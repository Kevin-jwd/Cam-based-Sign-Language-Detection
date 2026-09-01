"""
MediaPipe 3D 관절 좌표(x, y, z) + 3D 공간 각도(15개)를 결합한
78차원 초고속 병렬 추출 스크립트.
"""

import os
import glob
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DATASET_ROOT = "assets/data/ASL_SemCom"
OUTPUT_CSV = "assets/csv/asl_landmarks.csv"

# C++ 백엔드 로그 차단
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 전역 프로세스 전용 MediaPipe 객체
hands_detector = None


def init_worker():
    """각 CPU 코어(프로세스)별로 MediaPipe 객체를 독립 초기화합니다."""
    global hands_detector
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )


def extract_78d_features(landmarks_21):
    """
    MediaPipe 21개 3D 관절 좌표 (x, y, z) + 3D 관절 각도 (15개) 추출
    반환: 63개 정규화 3D 좌표 + 15개 3D 각도 = 총 78차원 피처 Vector
    """
    # 1. (21, 3) 3D 좌표 배열 추출
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_21])

    # 2. 위치 정규화 (손목 0번 관절 기준)
    rel_pts = pts - pts[0]

    # 3. 스케일 정규화 (3D 거리 기준)
    max_val = np.max(np.abs(rel_pts))
    norm_pts = rel_pts / max_val if max_val > 0 else rel_pts

    # 4. 3D 공간 상의 관절 각도(Angle) 15개 계산
    joint_triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),        # 엄지
        (0, 5, 6), (5, 6, 7), (6, 7, 8),        # 검지
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # 중지
        (0, 13, 14), (13, 14, 15), (14, 15, 16),# 약지
        (0, 17, 18), (17, 18, 19), (18, 19, 20) # 새끼
    ]
    
    angles = []
    for p1, p2, p3 in joint_triplets:
        v1 = norm_pts[p1] - norm_pts[p2]
        v2 = norm_pts[p3] - norm_pts[p2]
        
        # 3D 내적을 통한 공간 각도 계산
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(np.arccos(cos_angle))

    # 5. 최종 피처 결합 (63개 3D 좌표 + 15개 3D 각도 = 총 78차원)
    return np.hstack([norm_pts.flatten(), np.array(angles)])


def process_single_image(args):
    """단일 이미지 처리 워커 함수"""
    img_path, label = args
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    features = extract_78d_features(results.multi_hand_landmarks[0].landmark)
    row = features.tolist()
    row.append(label)
    return row


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 1. 처리할 이미지 작업 리스트 구축
    tasks = []
    sub_dirs = ["Train", "Test", "train", "test"]
    target_dirs = [os.path.join(DATASET_ROOT, d) for d in sub_dirs if os.path.exists(os.path.join(DATASET_ROOT, d))]
    if not target_dirs:
        target_dirs = [DATASET_ROOT]

    for target_dir in target_dirs:
        classes = sorted(os.listdir(target_dir))
        for label in classes:
            class_dir = os.path.join(target_dir, label)
            if not os.path.isdir(class_dir):
                continue

            image_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.png"))
            for img_path in image_paths:
                tasks.append((img_path, label))

    num_workers = os.cpu_count() or 4
    print(f"[INFO] Found {len(tasks)} total images.")
    print(f"[INFO] Starting 78D Parallel Extraction with {num_workers} CPU cores...")

    # 2. 멀티프로세싱 병렬 처리
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks, chunksize=100), total=len(tasks)))

    # None(손 미검출) 제거
    data_list = [r for r in results if r is not None]

    # 3. CSV 저장 헤더 구성 (x_i, y_i, z_i 총 63개 + angle_j 15개)
    feature_cols = []
    for i in range(21):
        feature_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    for j in range(15):
        feature_cols.append(f"angle_{j}")
    feature_cols.append("label")

    df = pd.DataFrame(data_list, columns=feature_cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Extraction Finished! Saved {len(df)} samples with 78 features to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()