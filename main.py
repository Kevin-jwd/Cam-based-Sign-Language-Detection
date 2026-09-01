"""
main.py
78차원 피처 1D-CNN 실시간 ASL 인식 및 Ollama 연동 메인 파일
"""
import time
import cv2

from utils.asl_classifier import ASLClassifier
from utils.camera_manager import CameraManager
from utils.ollama_manager import OllamaManager
from utils.text_manager import TextManager
from utils.ui_drawer import UIDrawer

MODEL_PATH = "assets/models/asl_1dcnn.pth"
CLASSES_PATH = "assets/models/classes.json"
CONF_THRESHOLD = 0.70


def main():
  print("[INFO] Initializing Inference Pipeline...")

  try:
    classifier = ASLClassifier(
        model_path=MODEL_PATH, classes_path=CLASSES_PATH
    )
    camera = CameraManager(width=640, height=480)
  except Exception as e:
    print(e)
    return

  text_manager = TextManager(threshold_count=7, cooldown_frames=10)
  ollama_mgr = OllamaManager(model_name="qwen2.5:3b")

  print("[SUCCESS] Pipeline Ready.")
  print(" - Press 'c' to Clear Text")
  print(" - Press 'ENTER' or 'g' to Send Text to Ollama")
  print(" - Press 'q' or ESC to Exit")

  while True:
    ret, frame = camera.read()
    if not ret:
      time.sleep(0.01)
      continue

    # 1. 모델 추론 및 랜드마크 렌더링
    current_pred, current_conf = classifier.predict(frame)

    if current_conf >= CONF_THRESHOLD:
      text_manager.update(current_pred, current_conf)

    # 2. 텍스트 및 UI 오버레이
    assembled_text = text_manager.get_text()
    ollama_result = ollama_mgr.get_response()
    frame = UIDrawer.draw(
        frame, current_pred, current_conf, assembled_text, ollama_result
    )

    cv2.imshow("Jetson Orin Nano - ASL to Ollama", frame)

    # 3. 키 입력 이벤트 제어
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
      break
    elif key == ord("c"):
      text_manager.clear()
      ollama_mgr.clear()
    elif key in (13, ord("g")):  # ENTER(13) 또는 'g' 키로 Ollama 호출
      prompt = text_manager.get_text()
      if prompt:
        print(f"[INFO] Sending to Ollama: {prompt}")
        system_prompt = (
            "You are a helpful assistant for sign language interpretation. "
            "Convert the given raw fingerspelled characters or words into a"
            " complete, natural English sentence. Correct any spelling errors. "
            "Output ONLY the corrected single sentence."
        )
        ollama_mgr.generate_async(
            prompt_text=prompt, system_prompt=system_prompt
        )

  classifier.close()
  camera.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()