import os
import shutil
from pathlib import Path

# 현재 파일(tools/merge_dataset.py) 위치를 기준으로 프로젝트 루트 디렉터리 경로 계산
# tools/ -> 프로젝트 루트
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 소스(Kaggle) 및 타겟(ASL_SemCom Train) 디렉터리 경로
SOURCE_DIR = PROJECT_ROOT / "assets" / "data" / "Gesture Image Data"
TARGET_DIR = PROJECT_ROOT / "assets" / "data" / "ASL_SemCom" / "Train"

def merge_datasets():
    if not SOURCE_DIR.exists():
        print(f"[Error] 소스 경로를 찾을 수 없습니다: {SOURCE_DIR}")
        print("  -> assets/data/Gesture Image Data 폴더가 복사되었는지 확인해 주세요.")
        return

    if not TARGET_DIR.exists():
        print(f"[Error] 타겟 Train 경로를 찾을 수 없습니다: {TARGET_DIR}")
        print("  -> assets/data/ASL_SemCom/Train 폴더가 존재하는지 확인해 주세요.")
        return

    print("=== 데이터셋 병합 작업 시작 ===")
    print(f" - 소스 경로 : {SOURCE_DIR}")
    print(f" - 타겟 경로 : {TARGET_DIR}\n")

    total_copied = 0

    # Gesture Image Data 내의 각 클래스 폴더(A, B, C... 0, 1...) 순회
    for class_folder in SOURCE_DIR.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            target_class_dir = TARGET_DIR / class_name
            
            # 타겟 폴더가 없으면 새로 생성
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            copied_count = 0
            for img_path in class_folder.glob("*.*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # 중복 방지용 Prefix 부여
                    new_filename = f"kaggle_{img_path.name}"
                    dst_file_path = target_class_dir / new_filename
                    
                    shutil.copy2(img_path, dst_file_path)
                    copied_count += 1
                    
            total_copied += copied_count
            print(f"[{class_name}] -> {copied_count}개 이미지 복사 완료 ({target_class_dir.name} 폴더)")

    print(f"\n[성공] 총 {total_copied}개 이미지 병합이 완료되었습니다!")

if __name__ == "__main__":
    merge_datasets()