"""
utils/ollama_manager.py
Ollama 비동기 API 연동 모듈
"""
import requests
import threading

class OllamaManager:
    # api_url 기본값을 localhost -> 127.0.0.1 로 변경
    def __init__(self, model_name: str = "qwen3.5:0.8b", api_url: str = "http://127.0.0.1:11434/api/generate"):
        """
        Args:
            model_name (str): 'qwen3.5:0.8b', 'gemma3:4b' 등
            api_url (str): Ollama 로컬 API 주소 (127.0.0.1 권장)
        """
        self.model_name = model_name
        self.api_url = api_url
        self.response_text = ""
        self.is_processing = False

    def generate_async(self, prompt_text: str, system_prompt: str = ""):
        if self.is_processing or not prompt_text.strip():
            return

        self.is_processing = True
        self.response_text = "Thinking..."
        
        thread = threading.Thread(
            target=self._request_ollama, 
            args=(prompt_text, system_prompt),
            daemon=True
        )
        thread.start()

    def _request_ollama(self, prompt_text: str, system_prompt: str):
        payload = {
            "model": self.model_name,
            "prompt": prompt_text,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                self.response_text = response.json().get("response", "").strip()
            else:
                self.response_text = f"[Error {response.status_code}] Request Failed"
        except Exception as e:
            self.response_text = f"[Connection Error] {str(e)}"
        finally:
            self.is_processing = False

    def get_response(self) -> str:
        return self.response_text

    def clear(self):
        self.response_text = ""
        self.is_processing = False