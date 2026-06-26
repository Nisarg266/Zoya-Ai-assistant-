# Zoya — Desktop AI Assistant

An advanced, modular Windows desktop AI assistant inspired by JARVIS.

Zoya is designed to be a highly modular, SOLID-compliant assistant that interacts natively with the Windows OS and understands natural language using Google's Gemini LLM.

---

## 🧠 Current Capabilities

Zoya is being built **one module at a time**. The following modules have been completed:

### 1. Desktop Automation Module
A comprehensive suite of tools that allows Zoya to control your computer natively. It acts as a set of plugins that the LLM can use.
- **Keyboard & Mouse Simulation**: Typing, hotkeys, clicking, scrolling, dragging.
- **Window Management**: Listing, focusing, minimizing, maximizing, closing windows.
- **Filesystem Operations**: Reading files, searching, listing directories safely.
- **Process Management**: Launching applications and terminating background processes.
- **System Controls**: Adjusting brightness, controlling master volume, handling clipboard data, and performing power actions.

### 2. Gemini Integration (The "Brain")
The core LLM engine that powers Zoya's intelligence.
- Uses the official `google-genai` SDK with `gemini-2.5-pro` (configurable).
- Automatically parses Zoya's automation tools (using Pydantic schemas) into Gemini Function Declarations.
- Features a **ReAct loop** (Reason + Act) allowing Zoya to receive a command, call local tools to gather info or perform actions, and formulate a natural language response.

### 3. Voice Input Module
Continuous, asynchronous **Speech-to-Text** so Zoya can listen and be spoken to.
- Powered by **Faster-Whisper** (CTranslate2-backed Whisper) running each transcription in a worker thread.
- Continuous microphone capture via `sounddevice`, bridged into the `asyncio` loop through a queue.
- Energy-based **Voice-Activity Detection** segments raw audio into utterances (silence-aware, with min/max duration guards).
- Supports **English (en), Hindi (hi), and Gujarati (gu)**; auto-detect or force a language. Detected speech outside the allowed set is discarded.
- Recognised text is handed to the Gemini Brain via `brain.chat()`. *(Text-to-Speech is intentionally not implemented yet.)*
- Fully self-contained: adds **no changes to core files** and imports cleanly even when the optional audio stack is absent.

---

## 📁 Project Structure

```text
Zoya-Ai-Assistant/
├── .env.example            # Environment variables (Gemini API keys, feature flags)
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.yaml       # Tunable automation defaults (e.g., typing speed)
├── src/zoya/
│   ├── core/               # AppConfig, logging, exceptions (shared by all modules)
│   ├── automation/         # The Desktop Automation tools & controllers
│   │   ├── controllers/    # Low-level, synchronous, one-job actors (e.g. keyboard.py)
│   │   ├── tools/          # ITool plugins & registry for LLM function calling
│   │   └── schemas.py      # Strict Pydantic parameter validation for every tool
│   ├── llm/                # Gemini Integration (The Brain)
│   │   ├── client.py       # Async Gemini API wrapper
│   │   ├── facade.py       # ZoyaBrain class (handles the tool execution loop)
│   │   └── function_tools.py # Bridges Pydantic schemas with Gemini
│   └── voice/              # Voice Input Module (Speech-to-Text)
│       ├── config.py       # VoiceSettings + settings.yaml loader (self-contained)
│       ├── capture.py      # Async microphone capture (sounddevice)
│       ├── transcriber.py  # Async Faster-Whisper wrapper (en/hi/gu)
│       ├── listener.py     # Continuous VAD utterance segmentation
│       └── pipeline.py     # VoiceInput facade + brain.chat() integration
└── scripts/
    ├── demo_gemini.py      # Interactive chat script to test Zoya's brain
    └── demo_voice.py       # Interactive mic + STT demo (with optional Brain)
```

---

## 🛠️ Architecture Highlights

Built with production-grade engineering principles in mind:
- **Dependency Inversion (DIP)**: Tools depend only on controller instances injected at construction time. The LLM interacts strictly with the `ITool` protocol.
- **Open/Closed Principle (OCP)**: Adding new capabilities is as easy as writing a new `XxxTool` class and registering it. The LLM immediately knows how to use it without touching core code!
- **Strict Validation**: All tools use `Pydantic` `extra="forbid"`. If the LLM hallucinates an argument, it is caught instantly at the boundary.
- **Safety First**: Destructive system actions (Restart/Shutdown) require explicit boolean confirmation flags. Files are deleted using `send2trash` (recycle bin).

---

## 🚀 Getting Started

### 1. Installation
```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt

# Setup your configuration
copy .env.example .env
```

### 2. Configuration
Open the newly created `.env` file and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

### 3. Run the Assistant
Test Zoya's brain interactively via the terminal:
```powershell
python scripts\demo_gemini.py
```
*Try asking it: "What's the weather like in Tokyo?" to see it automatically use its local tools to figure it out!*

---
*Developed with modern Python, AsyncIO, and SOLID architecture principles.*