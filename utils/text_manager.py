"""
utils/text_manager.py
실시간 수어 인식 결과의 디바운싱(Debouncing) 및 문장 조립 관리 모듈
"""

class TextManager:
    def __init__(self, threshold_count: int = 7, cooldown_frames: int = 10):
        """
        Args:
            threshold_count (int): 동일 수어가 몇 프레임 연속 유지되어야 글자로 확정할지 기준
            cooldown_frames (int): 글자 입력 후 다음 입력까지 대기할 쿨다운 프레임 수
        """
        self.text = ""
        self.last_pred = None
        self.consecutive_count = 0
        self.threshold_count = threshold_count
        self.cooldown_counter = 0
        self.cooldown_frames = cooldown_frames

    def update(self, pred_class: str, confidence: float = 1.0) -> bool:
        """
        매 프레임 추론 결과를 전달받아 디바운싱 후 문장에 추가합니다.
        
        Returns:
            bool: 새로운 글자가 추가된 순간 True 반환
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        if pred_class == self.last_pred:
            self.consecutive_count += 1
        else:
            self.last_pred = pred_class
            self.consecutive_count = 1

        if self.consecutive_count >= self.threshold_count:
            self._process_character(pred_class)
            self.consecutive_count = 0
            self.cooldown_counter = self.cooldown_frames
            return True

        return False

    def _process_character(self, char: str):
        char_upper = str(char).upper()

        if char_upper in ["SPACE", "BLANK"]:
            self.text += " "
        elif char_upper in ["DEL", "DELETE", "BACKSPACE"]:
            self.delete_last()
        elif char_upper == "CLEAR":
            self.clear()
        else:
            self.text += str(char)

    def add_space(self):
        self.text += " "

    def delete_last(self):
        if len(self.text) > 0:
            self.text = self.text[:-1]

    def clear(self):
        self.text = ""
        self.last_pred = None
        self.consecutive_count = 0
        self.cooldown_counter = 0

    def get_text(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text