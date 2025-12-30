# Push-to-Talk Speech-to-Text Tool

A local speech-to-text tool that records audio while you hold a key, transcribes it using OpenAI's Whisper model, and automatically types the result into your active window.

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed and available in your system PATH

### Install Python Dependencies

Run the following commands in your terminal:

```bash
pip install openai-whisper
pip install pynput
pip install sounddevice
pip install numpy
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### FFmpeg Installation

**Windows:**
- Download from https://ffmpeg.org/download.html
- Extract and add to PATH, or use: `choco install ffmpeg` (if you have Chocolatey)

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg  # Debian/Ubuntu
# or
sudo yum install ffmpeg      # CentOS/RHEL
```

## Usage

1. Run the script:
   ```bash
   python main.py
   ```

2. The script will load the Whisper model (this takes a moment on first run).

3. **Press and hold** the Right Shift key (or F8 if you modify the code) to start recording.

4. **Release** the key to stop recording, transcribe, and automatically type the text.

5. Press **Esc** to exit the program.

## Configuration

To change the trigger key, edit `main.py` and modify the `trigger_key` variable in the `main()` function:

```python
trigger_key = Key.shift_r  # Right Shift
# or
trigger_key = Key.f8       # F8 key
```

## Features

- ✅ Push-to-talk recording (hold key to record)
- ✅ Local Whisper transcription (no internet required)
- ✅ Automatic typing into active window
- ✅ Key repeat handling (prevents restarting on key repeats)
- ✅ Silence detection (skips empty/too-short audio)
- ✅ Non-blocking threading (smooth operation)
- ✅ Model loaded once at startup (fast subsequent transcriptions)

## Troubleshooting

- **No audio recorded**: Check your microphone permissions and default audio input device
- **Model download slow**: The first run downloads the model (~150MB for base model)
- **Typing not working**: Ensure the target application accepts keyboard input
- **Permission errors (macOS)**: Grant accessibility permissions to Terminal/Python in System Preferences

