"""
utils/ollama_manager.py
Ollama 비동기 API 연동, 입력 전처리 및 1줄/프로토콜 응답 제어 모듈
"""
import re
import requests
import threading
import traceback


def preprocess_fingerspelling(text: str) -> str:
    """카메라 수어 인식 시 발생하는 연속 중복 알파벳/단어 정제

    E.g., 'AAAAVR' -> 'AVR', 'AVVR' -> 'AVR', 'AAUUUDD' -> 'AUD'
    """
    if not text:
        return ""
    # 연속된 동일 문자(2회 이상)를 1개로 압축
    cleaned = re.sub(r"(.)\1+", r"\1", text.strip().upper())
    return cleaned


class OllamaManager:

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        api_url: str = "http://127.0.0.1:11434/api/chat",
    ):
        """OllamaManager 초기화

        Args:
            model_name (str): 사용할 Ollama 모델명 (기본값: 'qwen2.5:3b')
            api_url (str): Ollama Chat API 엔드포인트
        """
        self.model_name = model_name
        self.api_url = api_url
        self.response_text = ""
        self.is_processing = False

    def generate_async(self, prompt_text: str, system_prompt: str = ""):
        """비동기 방식으로 Ollama 모델에 답변을 요청 (스레드 생성)"""
        if self.is_processing or not prompt_text.strip():
            return

        # 입력 텍스트 정제 (AVVR -> AVR)
        clean_prompt = preprocess_fingerspelling(prompt_text)

        self.is_processing = True
        self.response_text = "Thinking..."

        print("\n" + "-" * 50)
        print(
            f"[OLLAMA REQ] Raw: '{prompt_text}' -> Cleaned: '{clean_prompt}' |"
            f" Model: '{self.model_name}'"
        )

        thread = threading.Thread(
            target=self._request_ollama,
            args=(clean_prompt, system_prompt),
            daemon=True,
        )
        thread.start()

    def _request_ollama(self, prompt_text: str, system_prompt: str):
        # 1줄 한국어 요약 출력용 기본 시스템 프롬프트
        default_sys = (
            "너는 수어 단어 및 기술 키워드를 해석하는 AI다.\n"
            "사용자가 'AVR', 'AUD' 등 키워드를 입력하면 해당 의미를 한국어 딱"
            " 1문장(한 줄)으로만 설명해라.\n\n"
            "[작성 규칙]\n"
            "1. 줄바꿈, 인사말, 영어 설명, 부연설명 없이 오직 한국어 1문장만"
            " 출력할 것.\n"
            "2. 예시: 'AVR' -> 'AVR은 임베디드 시스템 및 아두이노 등에 주로"
            " 사용되는 8비트 RISC 구조의 Atmel 마이크로컨트롤러입니다.'"
        )
        sys_msg = system_prompt if system_prompt else default_sys

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "keep_alive": -1,  # VRAM 상주 (Cold Start 방지)
            "options": {
                "num_predict": 64,  # 한 줄 출력을 위한 토큰 제한
                "temperature": 0.0,  # 환각 방지 및 일관성 극대화
                "stop": ["\n"],  # 줄바꿈 생성 시 즉시 응답 종료 (속도 향상)
            },
        }

        try:
            print(f"[OLLAMA INF] POST -> {self.api_url}")

            # 프록시 우회 및 연결/응답 타임아웃 설정 (10초, 120초)
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=(10, 120),
                proxies={"http": None, "https": None},
            )

            print(f"[OLLAMA INF] HTTP Status Code: {response.status_code}")

            if response.status_code == 200:
                res_json = response.json()
                msg_content = (
                    res_json.get("message", {}).get("content", "").strip()
                )

                if msg_content:
                    self.response_text = msg_content
                else:
                    self.response_text = "[Ollama Warning] Empty Response"

                print(f"[OLLAMA SUCCESS] Result: {self.response_text}")
                print("-" * 50 + "\n")
            else:
                self.response_text = f"[Error {response.status_code}] Check Terminal"
                print(f"[OLLAMA FAIL] Status: {response.status_code}")
                print(f"[OLLAMA FAIL BODY]: {response.text}")
                print("-" * 50 + "\n")

        except Exception as e:
            self.response_text = "[Connection Error] Check Terminal"
            print("\n================ [OLLAMA EXCEPTION LOG] ================")
            print(f"Error Type   : {type(e).__name__}")
            print(f"Error Message: {e}")
            print("\n[Traceback Details]:")
            traceback.print_exc()
            print("========================================================\n")

        finally:
            self.is_processing = False

    def get_response(self) -> str:
        """최신 답변 반환"""
        return self.response_text

    def clear(self):
        """답변 상태 초기화"""
        self.response_text = ""
        self.is_processing = False