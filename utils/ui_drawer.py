"""
utils/ui_drawer.py
OpenCV 텍스트 및 하단 검은색 배경 오버레이 UI 모듈
"""
import cv2


class UIDrawer:

  @staticmethod
  def draw(
      frame,
      current_pred: str,
      current_conf: float,
      assembled_text: str,
      ollama_result: str,
  ):
    # 1. 상단: 현재 감지된 실시간 수어 정보
    if current_pred:
      label_text = f"Current: {current_pred} ({current_conf*100:.1f}%)"
      cv2.putText(
          frame,
          label_text,
          (20, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          (0, 255, 0),
          2,
      )

    # 2. 하단: 누적 텍스트 및 Ollama 응답 UI (하단 검은색 배경 100px)
    cv2.rectangle(frame, (0, 380), (640, 480), (20, 20, 20), -1)

    cv2.putText(
        frame,
        f"Text  : {assembled_text}",
        (15, 415),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Ollama: {ollama_result}",
        (15, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    return frame