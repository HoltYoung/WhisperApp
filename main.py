#!/usr/bin/env python3
"""
Push-to-Talk Speech-to-Text Tool using OpenAI Whisper
Records audio while holding a key, transcribes on release, and types the result.
"""

import threading
import queue
import time
import numpy as np
import sounddevice as sd
import whisper
from pynput import keyboard
from pynput.keyboard import Key, KeyCode, Listener
import sys
import tkinter as tk
from tkinter import ttk


class PushToTalkSTT:
    def __init__(self, trigger_key=Key.ctrl_r, model_size="base", sample_rate=16000):
        """
        Initialize the Push-to-Talk Speech-to-Text tool.
        
        Args:
            trigger_key: The key to hold for recording (default: Right Shift)
            model_size: Whisper model size (default: "base")
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.trigger_key = trigger_key
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.transcription_thread = None
        self.audio_buffer = []
        self.stream = None
        self.recording_lock = threading.Lock()
        
        # Create overlay widget
        self._create_overlay()
        
        # Load Whisper model once at startup
        print(f"Loading Whisper model ({model_size})... This may take a moment.")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")
        
        # Keyboard controller for typing
        self.keyboard_controller = keyboard.Controller()
    
    def _create_overlay(self):
        """Create a small always-on-top overlay window."""
        self.root = tk.Tk()
        self.root.title("Push-to-Talk Status")
        self.root.attributes("-topmost", True)  # Always on top
        try:
            self.root.attributes("-alpha", 0.85)  # Slight transparency (may not work on all systems)
        except:
            pass
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Position in top-right corner
        screen_width = self.root.winfo_screenwidth()
        self.root.geometry(f"150x60+{screen_width - 160}+10")
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="● Ready",
            font=("Arial", 12, "bold"),
            fg="gray",
            bg="black"
        )
        self.status_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.root.configure(bg="black")
        
        # Make window update periodically
        def periodic_update():
            try:
                self.root.update_idletasks()
                self.root.update()
                self.root.after(50, periodic_update)  # Update every 50ms
            except:
                pass
        
        # Start periodic updates
        self.root.after(50, periodic_update)
        
        # Update thread for GUI mainloop
        def update_gui():
            try:
                self.root.mainloop()
            except:
                pass
        
        gui_thread = threading.Thread(target=update_gui, daemon=True)
        gui_thread.start()
    
    def _update_status(self, status, color):
        """Update the overlay status display (thread-safe)."""
        try:
            # Use after() for thread-safe GUI updates
            self.root.after(0, lambda: self.status_label.config(text=status, fg=color))
        except:
            pass  # Ignore errors if window is closed
        
    def on_key_press(self, key):
        """Handle key press events."""
        # Check if the trigger key is pressed
        # Handle both Key objects and character keys (KeyCode)
        key_matches = False
        if isinstance(self.trigger_key, str):
            # Character key - check if it matches
            if hasattr(key, 'char') and key.char == self.trigger_key:
                key_matches = True
        else:
            # Special key (Key object)
            if key == self.trigger_key:
                key_matches = True
        
        if key_matches:
            with self.recording_lock:
                if not self.is_recording:
                    self.is_recording = True
                    self.audio_buffer = []
                    print("Recording started... (hold key to record)")
                    self._update_status("● RECORDING", "red")
                    
                    # Start recording in a separate thread
                    self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
                    self.recording_thread.start()
        # Ignore key repeats - the flag prevents restarting
    
    def on_key_release(self, key):
        """Handle key release events."""
        # Handle both Key objects and character keys (KeyCode)
        key_matches = False
        if isinstance(self.trigger_key, str):
            # Character key - check if it matches
            if hasattr(key, 'char') and key.char == self.trigger_key:
                key_matches = True
        else:
            # Special key (Key object)
            if key == self.trigger_key:
                key_matches = True
        
        if key_matches:
            with self.recording_lock:
                if self.is_recording:
                    self.is_recording = False
                    print("Recording stopped. Processing...")
                    self._update_status("● Processing...", "yellow")
                    
                    # Wait a moment for the recording thread to finish capturing
                    time.sleep(0.2)
                    
                    # Process the recorded audio
                    if self.audio_buffer:
                        self._process_audio()
                    else:
                        print("No audio captured.")
                        self._update_status("● Ready", "gray")
    
    def _record_audio(self):
        """Record audio in a separate thread while is_recording is True."""
        try:
            # Use callback-based recording for better reliability
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                if self.is_recording:
                    # Make a copy of the data to avoid issues
                    self.audio_buffer.append(indata.copy().flatten())
            
            # Start the stream with callback
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=audio_callback
            )
            
            self.stream.start()
            
            # Small delay to ensure stream is ready
            time.sleep(0.1)
            
            # Keep recording while flag is True
            while self.is_recording:
                time.sleep(0.05)  # Small sleep to prevent busy waiting
            
            # Stop the stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
        except Exception as e:
            print(f"Error during recording: {e}")
            import traceback
            traceback.print_exc()
            with self.recording_lock:
                self.is_recording = False
            self._update_status("● Error", "red")
    
    def _process_audio(self):
        """Process recorded audio and transcribe it."""
        # Run transcription in a separate thread to avoid blocking
        transcription_thread = threading.Thread(target=self._transcribe_and_type, daemon=True)
        transcription_thread.start()
    
    def _transcribe_and_type(self):
        """Transcribe audio and type the result."""
        try:
            # Concatenate all audio chunks
            if not self.audio_buffer:
                print("No audio buffer to process.")
                self._update_status("● Ready", "gray")
                return
            
            audio_data = np.concatenate(self.audio_buffer)
            
            # Debug info
            print(f"Audio length: {len(audio_data) / self.sample_rate:.2f} seconds")
            print(f"Audio shape: {audio_data.shape}")
            
            # Check for silence/empty audio
            # Calculate RMS (Root Mean Square) to detect silence
            rms = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.abs(audio_data).max()
            print(f"RMS: {rms:.6f}, Max amplitude: {max_amplitude:.6f}")
            
            # More lenient thresholds
            silence_threshold = 0.001  # Lowered threshold
            min_duration = 0.2  # Minimum 0.2 seconds
            
            if len(audio_data) < self.sample_rate * min_duration:
                print(f"Audio too short ({len(audio_data) / self.sample_rate:.2f}s), skipping.")
                self._update_status("● Too short", "orange")
                time.sleep(1)
                self._update_status("● Ready", "gray")
                return
            
            if rms < silence_threshold and max_amplitude < 0.01:
                print(f"Audio appears to be silence (RMS: {rms:.6f}), skipping.")
                self._update_status("● Silence", "orange")
                time.sleep(1)
                self._update_status("● Ready", "gray")
                return
            
            # Normalize audio to [-1, 1] range if needed
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Ensure audio is in the right format for Whisper (float32, mono)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Transcribe using Whisper
            print("Transcribing...")
            result = self.model.transcribe(audio_data, language=None, fp16=False)
            text = result["text"].strip()
            
            if text:
                print(f"Transcribed: {text}")
                # Type the text
                self._type_text(text)
                self._update_status("● Ready", "gray")
            else:
                print("No text transcribed.")
                self._update_status("● No text", "orange")
                time.sleep(1)
                self._update_status("● Ready", "gray")
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            self._update_status("● Error", "red")
            time.sleep(1)
            self._update_status("● Ready", "gray")
    
    def _type_text(self, text):
        """Type the text into the active window."""
        try:
            # Small delay to ensure focus is on the target window
            time.sleep(0.1)
            
            # Type the text character by character
            for char in text:
                self.keyboard_controller.type(char)
                time.sleep(0.01)  # Small delay between characters for reliability
            
            print("Text typed successfully.")
        except Exception as e:
            print(f"Error typing text: {e}")
    
    def start(self):
        """Start the keyboard listener."""
        print(f"\nPush-to-Talk Speech-to-Text Tool")
        if isinstance(self.trigger_key, str):
            key_name = self.trigger_key.upper()
        elif self.trigger_key == Key.ctrl_r:
            key_name = "Right Ctrl"
        elif self.trigger_key == Key.alt_r:
            key_name = "Right Alt"
        elif self.trigger_key == Key.f8:
            key_name = "F8"
        elif self.trigger_key == Key.shift_r:
            key_name = "Right Shift"
        else:
            key_name = str(self.trigger_key)
        print(f"Press and hold '{key_name}' to record, release to transcribe and type.")
        print("Press 'Esc' to exit.\n")
        
        # Test audio devices
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            print(f"Default input device: {default_input['name']}")
        except Exception as e:
            print(f"Warning: Could not query audio devices: {e}")
        
        def on_press(key):
            self.on_key_press(key)
            # Allow Esc to exit
            if key == Key.esc:
                print("\nExiting...")
                with self.recording_lock:
                    self.is_recording = False
                if self.stream:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
                return False
        
        def on_release(key):
            self.on_key_release(key)
        
        # Start the keyboard listener
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


def main():
    """Main entry point."""
    # You can change the trigger key here:
    # Key.ctrl_r = Right Ctrl (default - easy to press, won't type anything)
    # Key.alt_r = Right Alt
    # Key.scroll_lock = Scroll Lock
    # Key.insert = Insert key
    # Key.f8 = F8 key (requires Fn on many laptops)
    # Or use a character key like 'r', 't', etc. (but it will type that character)
    
    trigger_key = Key.ctrl_r  # Right Ctrl - hold to record, won't interfere with typing
    
    try:
        app = PushToTalkSTT(trigger_key=trigger_key, model_size="base")
        app.start()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

