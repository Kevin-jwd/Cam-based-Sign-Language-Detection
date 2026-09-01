"""
utils/camera_manager.py
V4L2 백엔드 기반 카메라 장치 제어 모듈 (인덱스 0 -> 1 폴백 로직 포함)
"""
import cv2


class CameraManager:

  def __init__(self, width: int = 640, height: int = 480):
    self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not self.cap.isOpened():
      self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
      if not self.cap.isOpened():
        raise RuntimeError("[FATAL] No video camera found!")

    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  def read(self):
    return self.cap.read()

  def release(self):
    if self.cap and self.cap.isOpened():
      self.cap.release()