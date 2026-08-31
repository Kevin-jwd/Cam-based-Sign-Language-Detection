# 🤟 Cam-based Sign Language Detection with Ollama

카메라로 인식된 수어 알파벳(Fingerspelling) 및 구어체 텍스트를 로컬 LLM(Ollama `qwen2.5:3b`)과 연동하여 한국어 설명 및 제어 프로토콜로 변환하는 프로젝트입니다.

---

## 📌 주요 기능

- **연속 중복 인식 정제**: 카메라 프레임 특성상 발생하는 중복 문자를 자동 압축 (예: `AAAAVR` → `AVR`)
- **로컬 LLM 연동**: Ollama 비동기 API 통신을 통해 화면 멈춤(Freezing) 없는 실시간 처리
- **저지연(Low Latency) 응답**: 1줄 출력 제약 및 `stop` 옵션으로 1초 이내 빠른 결과 반환

---

## 🛠️ 요구 사항 (Prerequisites)

- **OS**: Ubuntu (Jetson Orin Nano 권장) / Windows / macOS
- **Python**: 3.8 이상
- **Ollama**: [Ollama 공식 웹사이트](https://ollama.com)를 통해 설치 필요

---

## 🚀 빠른 시작 (Quick Start)

### 1. 저장소 클론 및 이동
```bash
git clone [https://github.com/YOUR_USERNAME/Cam-based-Sign-Language-Detection.git](https://github.com/YOUR_USERNAME/Cam-based-Sign-Language-Detection.git)
cd Cam-based-Sign-Language-Detection
```

### 2. 가상환경 구축 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ollama 모델 다운로드
Ollama 서버가 실행 중인 상태에서 아래 명령어로 모델을 다운로드합니다.
```bash
ollama pull qwen2.5:3b
```

### 4. 메인 프로그램 실행
```bash
python main.py
```

---

## 💡 테스트 방법 (Usage)

1. 프로그램 실행 후 카메라 화면을 통해 수어를 입력하거나 텍스트를 전달합니다.
2. `AAVR`, `AVVR` 등 중복된 알파벳이 입력되더라도 내부 전처리를 거쳐 `AVR`로 자동 정제됩니다.
3. `Enter` 키 입력 시 Ollama 모델이 해당 단어를 추론하여 한 줄로 된 한국어 설명을 반환합니다.

---

## 📂 프로젝트 구조

```text
├── main.py                   # 메인 실행 파일 및 GUI/카메라 루프
├── utils/
│   ├── ollama_manager.py     # Ollama 비동기 API 통신 및 텍스트 전처리
│   └── text_manager.py       # 수어 인식 텍스트 관리 모듈
├── requirements.txt          # 의존성 패키지 목록
└── README.md                 # 프로젝트 가이드 문서
```