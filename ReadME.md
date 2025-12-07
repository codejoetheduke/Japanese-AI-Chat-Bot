# **Speech-to-Speech AI Pipeline (Whisper ASR + Ollama LLM + Edge TTS)**

![alt text](image.jpg)

A real-time, multilingual, **speech-to-speech AI assistant** that listens, understands, and responds with natural voice output — all running **locally**.

This project combines:

- **Automatic Speech Recognition (ASR)** using Whisper & specialized models
- **Local LLM reasoning** using Ollama
- **High-quality Text-to-Speech (TTS)** using Microsoft Edge TTS (free, no API key needed)

Supports **20+ languages**, including English, Japanese, Chinese, Spanish, French, Arabic, Korean, and more.

---

## 🚀 **Features**

### 🎤 **1. Speech-to-Text (ASR)**

- Uses **OpenAI Whisper** and specialized models
- Japanese uses **kotoba-tech/kotoba-whisper-v1.0** for superior accuracy
- Supports automatic normalization, resampling, and long audio segments

### 🤖 **2. LLM Reasoning (Ollama)**

- Integrates seamlessly with **Ollama’s local LLMs**
- Supports any model (Llama, Qwen, Mistral, Phi, etc.)
- Keeps **conversation history** for natural dialogue

### 🔊 **3. Text-to-Speech (Edge TTS)**

- Uses **Microsoft Edge Neural Voices**
- Completely free, high-quality, and supports 90+ voices
- No API keys or cloud access required

### 🌐 **4. Multilingual Support**

Preconfigured for high-quality voices + models:

```
English, Japanese, Chinese, Spanish, French, German,
Italian, Portuguese, Korean, Russian, Arabic, Hindi,
Polish, Turkish, Dutch, Czech, Hungarian, Swedish,
Norwegian, Finnish
```

---

## 📦 **Installation**

### 1. Install Python packages

```bash
pip install torch transformers sounddevice scipy numpy requests edge-tts
```

(Optional for MP3 → WAV conversion and playback)

```bash
pip install pydub pygame
```

---

### 2. Install and run Ollama

Download from: [https://ollama.com](https://ollama.com)

Start server:

```bash
ollama serve
```

Pull the model you want to use (example):

```bash
ollama pull llama3.2:3b
```

---

## 🔧 **Configuration**

Inside the script:

```python
LANGUAGE = "japanese"
OLLAMA_MODEL = "gpt-oss:120b-cloud"
CUSTOM_VOICE = None
```

You can switch:

- **LANGUAGE** → any from `SUPPORTED_LANGUAGES`
- **OLLAMA_MODEL** → any model installed in Ollama
- **CUSTOM_VOICE** → any Edge TTS voice name (optional)

List voices for a language:

```python
SpeechPipelineEdgeTTS.print_available_voices(language_filter="ja")
```

---

## ▶️ **Usage**

### **Start the full pipeline**

```bash
python app.py
```

The program will:

1. Record 5 seconds of your speech
2. Transcribe it using Whisper
3. Send text to the Ollama LLM
4. Convert the reply to natural speech
5. Play the audio output

You can continue chatting in a loop.

---

## 📁 **Project Structure**

```
├── SpeechPipelineEdgeTTS
│   ├── ASR (Whisper / Kotoba)
│   ├── Ollama LLM Chat Interface
│   ├── Edge TTS Voice Synthesis
│   ├── Microphone + Audio Playback
│   ├── Conversation Memory Handling
└── README.md
```

---

## 🧠 **How It Works**

### 1. **Record Microphone Audio**

Uses `sounddevice` for high-quality capture.

### 2. **Transcribe (Speech → Text)**

Runs a Whisper-based model optimized for the selected language.

### 3. **LLM Processing**

Sends text to Ollama with configurable temperature, memory, and model selection.

### 4. **Generate Natural Speech**

Converts the LLM output into speech using Edge TTS
(Saves MP3 → Converts to WAV → Plays audio)

---

## 🌍 **Supported Languages & Voices**

Each language maps to:

- Whisper language mode
- Specialized ASR model
- Best Edge TTS neural voice

You can customize these through the dictionary:

```python
SUPPORTED_LANGUAGES = {
    'japanese': {
        'whisper': 'japanese',
        'voice': 'ja-JP-NanamiNeural',
        'asr_model': 'kotoba-tech/kotoba-whisper-v1.0'
    }
}
```

---

## 🔄 **Conversation Memory**

Each interaction is stored:

```python
self.conversation_history.append({"role": "user", "content": text})
self.conversation_history.append({"role": "assistant", "content": ai_response})
```

Auto-clears old messages to avoid memory bloating.

Reset manually:

```python
pipeline.reset_conversation()
```

---

## 🗣 **Custom Voices**

Use any Edge TTS voice:

```python
CUSTOM_VOICE = "ja-JP-KeitaNeural"
```

Find all voices:

```python
SpeechPipelineEdgeTTS.print_available_voices()
```

---

## 🛠 Troubleshooting

### ❗ Ollama not detected

Make sure it's running:

```bash
ollama serve
```

### ❗ MP3/WAV playback not working

Install the fallback:

```bash
pip install pygame
```

### ❗ Whisper too slow

Switch to a smaller ASR model:

```python
'asr_model': 'openai/whisper-small'
```

---

## ⭐ **Future Improvements**

- Streaming ASR + Streaming TTS
- Realtime echo cancellation
- Web UI (Gradio / FastAPI)
- Hotword activation (“Hey Assistant…”)

---

## 📜 **License**

MIT License

---

## 👨‍💻 Author

**Duke Kojo Kongo** (_CodeJoe_)  
Data Scientist • AI Engineer • Builder of Intelligent Systems

---
