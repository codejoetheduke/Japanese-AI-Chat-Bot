"""
Speech-to-Speech AI Pipeline with Ollama LLM + Edge TTS
High-quality multilingual TTS using Microsoft Edge (free, no API key needed)
Requires: pip install torch transformers sounddevice scipy numpy requests edge-tts
"""

import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor,
    pipeline
)
import edge_tts
import asyncio
import requests
import warnings
warnings.filterwarnings('ignore')

# Supported languages with high-quality Edge TTS voices and specialized ASR models
SUPPORTED_LANGUAGES = {
    'english': {'whisper': 'english', 'voice': 'en-US-AriaNeural', 'asr_model': 'openai/whisper-base'},
    'english_male': {'whisper': 'english', 'voice': 'en-US-GuyNeural', 'asr_model': 'openai/whisper-base'},
    'spanish': {'whisper': 'spanish', 'voice': 'es-ES-ElviraNeural', 'asr_model': 'openai/whisper-base'},
    'french': {'whisper': 'french', 'voice': 'fr-FR-DeniseNeural', 'asr_model': 'openai/whisper-base'},
    'german': {'whisper': 'german', 'voice': 'de-DE-KatjaNeural', 'asr_model': 'openai/whisper-base'},
    'italian': {'whisper': 'italian', 'voice': 'it-IT-ElsaNeural', 'asr_model': 'openai/whisper-base'},
    'portuguese': {'whisper': 'portuguese', 'voice': 'pt-BR-FranciscaNeural', 'asr_model': 'openai/whisper-base'},
    'chinese': {'whisper': 'chinese', 'voice': 'zh-CN-XiaoxiaoNeural', 'asr_model': 'openai/whisper-small'},
    'japanese': {'whisper': 'japanese', 'voice': 'ja-JP-NanamiNeural', 'asr_model': 'kotoba-tech/kotoba-whisper-v1.0'},  # Japanese-specific model
    'korean': {'whisper': 'korean', 'voice': 'ko-KR-SunHiNeural', 'asr_model': 'openai/whisper-small'},
    'russian': {'whisper': 'russian', 'voice': 'ru-RU-SvetlanaNeural', 'asr_model': 'openai/whisper-base'},
    'arabic': {'whisper': 'arabic', 'voice': 'ar-SA-ZariyahNeural', 'asr_model': 'openai/whisper-small'},
    'hindi': {'whisper': 'hindi', 'voice': 'hi-IN-SwaraNeural', 'asr_model': 'openai/whisper-small'},
    'polish': {'whisper': 'polish', 'voice': 'pl-PL-ZofiaNeural', 'asr_model': 'openai/whisper-base'},
    'turkish': {'whisper': 'turkish', 'voice': 'tr-TR-EmelNeural', 'asr_model': 'openai/whisper-base'},
    'dutch': {'whisper': 'dutch', 'voice': 'nl-NL-ColetteNeural', 'asr_model': 'openai/whisper-base'},
    'czech': {'whisper': 'czech', 'voice': 'cs-CZ-VlastaNeural', 'asr_model': 'openai/whisper-base'},
    'hungarian': {'whisper': 'hungarian', 'voice': 'hu-HU-NoemiNeural', 'asr_model': 'openai/whisper-base'},
    'swedish': {'whisper': 'swedish', 'voice': 'sv-SE-SofieNeural', 'asr_model': 'openai/whisper-base'},
    'norwegian': {'whisper': 'norwegian', 'voice': 'nb-NO-PernilleNeural', 'asr_model': 'openai/whisper-base'},
    'finnish': {'whisper': 'finnish', 'voice': 'fi-FI-NooraNeural', 'asr_model': 'openai/whisper-base'},
}

class SpeechPipelineEdgeTTS:
    def __init__(self, language='english', 
                 ollama_model='llama3.2:3b',
                 ollama_url='http://localhost:11434',
                 custom_voice=None,  # Optional: use any Edge TTS voice
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the pipeline with Edge TTS (Microsoft's high-quality free TTS)
        
        Args:
            language: Language to use (see SUPPORTED_LANGUAGES)
            ollama_model: Ollama model name
            ollama_url: Ollama server URL
            custom_voice: Custom Edge TTS voice name (optional)
            device: Device to use for models
        """
        self.device = device
        self.language = language.lower()
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{language}' not supported. Choose from: {list(SUPPORTED_LANGUAGES.keys())}")
        
        print(f"Using device: {self.device}")
        print(f"Language: {language}")
        print(f"Ollama model: {ollama_model}")
        print("\n🔄 Loading models...")
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # 1. Speech-to-Text: Whisper or specialized model
        print("   Loading ASR model for speech-to-text...")
        self.asr_model_id = SUPPORTED_LANGUAGES[self.language]['asr_model']
        print(f"   Using model: {self.asr_model_id}")
        
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            max_new_tokens=256,  # Increased for Japanese/Chinese
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,  # Disable timestamps for better accuracy
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )
        
        # Set language for Whisper
        whisper_lang = SUPPORTED_LANGUAGES[self.language]['whisper']
        self.asr_pipe.model.config.forced_decoder_ids = self.asr_processor.get_decoder_prompt_ids(
            language=whisper_lang, 
            task="transcribe"
        )
        
        # 2. Text-to-Speech: Edge TTS
        print(f"   Setting up Edge TTS for {language}...")
        self.tts_voice = custom_voice or SUPPORTED_LANGUAGES[self.language]['voice']
        print(f"   Voice: {self.tts_voice}")
        
        print("✅ All models loaded successfully!\n")
        
        self.conversation_history = []
    
    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if not any(self.ollama_model in name for name in model_names):
                    print(f"   ⚠️  Warning: Model '{self.ollama_model}' not found in Ollama.")
                    print(f"   Available models: {model_names}")
                    print(f"   Run: ollama pull {self.ollama_model}")
                else:
                    print(f"   ✅ Ollama connected successfully!")
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Warning: Cannot connect to Ollama at {self.ollama_url}")
            print(f"   Make sure Ollama is running: ollama serve")
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        print(f"🎤 Recording for {duration} seconds...")
        print("   Speak now!")
        
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        print("   Recording finished!")
        return audio.flatten(), sample_rate
    
    def load_audio(self, file_path):
        """Load audio from file"""
        sample_rate, audio = read(file_path)
        
        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
            
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        return audio, sample_rate
    
    def speech_to_text(self, audio, sample_rate):
        """Convert speech to text using Whisper"""
        print(f"\n🎤 Step 1: Speech to Text (Whisper - {self.language})")
        
        try:
            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
            
            result = self.asr_pipe(audio)
            text = result["text"].strip()
            
            print(f"   Transcribed: '{text}'")
            return text
            
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    def process_with_ollama(self, text):
        """Process text through Ollama LLM"""
        print(f"\n🤖 Step 2: LLM Processing (Ollama - {self.ollama_model})")
        print(f"   Input: '{text}'")
        
        try:
            messages = self.conversation_history + [
                {"role": "user", "content": text}
            ]
            
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['message']['content'].strip()
                
                self.conversation_history.append({"role": "user", "content": text})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                if len(self.conversation_history) > 12:
                    self.conversation_history = self.conversation_history[-12:]
                
                print(f"   Response: '{ai_response[:150]}...'")
                return ai_response
            else:
                print(f"   Error: Ollama returned status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    async def _generate_speech(self, text, output_file):
        """Internal async function to generate speech"""
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(output_file)
    
    def text_to_speech(self, text, play_audio=True, save_path="output.wav"):
        """Convert text to speech using Edge TTS"""
        print(f"\n🔊 Step 3: Text to Speech (Edge TTS - {self.language})")
        print(f"   Voice: {self.tts_voice}")
        
        try:
            # Generate speech using Edge TTS (async) - save as MP3 first
            mp3_path = save_path.replace('.wav', '.mp3')
            asyncio.run(self._generate_speech(text, mp3_path))
            
            print(f"   Audio saved: {mp3_path}")
            
            # Convert MP3 to WAV and play
            if play_audio:
                try:
                    # Try using pydub + ffmpeg first
                    from pydub import AudioSegment
                    
                    print("   Converting MP3 to WAV...")
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio.export(save_path, format="wav")
                    
                    print("   Playing audio...")
                    sample_rate, audio_data = read(save_path)
                    
                    # Convert to float32
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_float = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    # Convert stereo to mono if needed
                    if len(audio_float.shape) > 1:
                        audio_float = audio_float.mean(axis=1)
                    
                    sd.play(audio_float, sample_rate)
                    sd.wait()
                    
                except (ImportError, Exception) as e:
                    # Fallback: Try pygame for MP3 playback (no ffmpeg needed)
                    try:
                        print("   Trying pygame for playback...")
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(mp3_path)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        
                        pygame.mixer.quit()
                        print("   ✅ Playback complete!")
                        
                    except ImportError:
                        print("   ⚠️  Audio playback not available")
                        print("   Install one of these:")
                        print("     Option 1: pip install pygame (easiest, no extra setup)")
                        print("     Option 2: pip install pydub + ffmpeg")
                        print(f"   Audio saved to {mp3_path} (you can play it manually)")
                    except Exception as e2:
                        print(f"   ⚠️  Playback error: {e2}")
                        print(f"   Audio saved to {mp3_path} (you can play it manually)")
            
            return mp3_path
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_pipeline(self, audio_input=None, use_microphone=True, 
                    record_duration=5, play_audio=True):
        """Run the complete speech-to-speech pipeline"""
        print("=" * 70)
        print(f"Starting Speech-to-Speech Pipeline ({self.language.upper()})")
        print("=" * 70)
        
        # Get audio input
        if use_microphone:
            audio, sample_rate = self.record_audio(duration=record_duration)
        else:
            audio, sample_rate = self.load_audio(audio_input)
        
        # Step 1: Speech to Text
        transcribed_text = self.speech_to_text(audio, sample_rate)
        if not transcribed_text:
            print("\n❌ Pipeline failed at speech-to-text stage")
            return None
        
        # Step 2: Process with Ollama
        ai_response = self.process_with_ollama(transcribed_text)
        if not ai_response:
            print("\n❌ Pipeline failed at LLM processing stage")
            return None
        
        # Step 3: Text to Speech
        audio_file = self.text_to_speech(ai_response, play_audio)
        if not audio_file:
            print("\n❌ Pipeline failed at text-to-speech stage")
            return None
        
        print("\n" + "=" * 70)
        print("✅ Pipeline completed successfully!")
        print("=" * 70)
        
        return {
            'transcript': transcribed_text,
            'ai_response': ai_response,
            'audio_file': audio_file
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
    
    @staticmethod
    async def list_voices():
        """List all available Edge TTS voices"""
        voices = await edge_tts.list_voices()
        return voices
    
    @staticmethod
    def print_available_voices(language_filter=None):
        """Print available voices, optionally filtered by language"""
        voices = asyncio.run(SpeechPipelineEdgeTTS.list_voices())
        
        if language_filter:
            voices = [v for v in voices if language_filter.lower() in v['Locale'].lower()]
        
        print(f"\nAvailable voices ({len(voices)}):")
        for v in voices:
            print(f"  {v['ShortName']}: {v['Locale']} - {v['Gender']}")


# Example usage
if __name__ == "__main__":
    # Configuration
    LANGUAGE = "japanese"  # Change to your preferred language
    OLLAMA_MODEL = "gpt-oss:120b-cloud"  # Change to your preferred model
    
    # Optional: Use a custom voice
    # Run: python script.py --list-voices to see all available voices
    CUSTOM_VOICE = None  # e.g., "ja-JP-KeitaNeural" for male Japanese voice
    
    # Initialize pipeline
    print("Initializing Speech-to-Speech Pipeline with Edge TTS...")
    print(f"Language: {LANGUAGE}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print("\nMake sure Ollama is running! (Run: ollama serve)\n")
    
    # Uncomment to see available voices:
    # SpeechPipelineEdgeTTS.print_available_voices(language_filter="ja")
    
    pipeline = SpeechPipelineEdgeTTS(
        language=LANGUAGE,
        ollama_model=OLLAMA_MODEL,
        custom_voice=CUSTOM_VOICE
    )
    
    # Run the pipeline
    print("\nReady! Press Enter to start recording...")
    input()
    
    result = pipeline.run_pipeline(
        use_microphone=True, 
        record_duration=5, 
        play_audio=True
    )
    
    # Continue conversation
    while True:
        print("\n\nContinue conversation? (Press Enter or type 'quit' to exit)")
        user_input = input()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        result = pipeline.run_pipeline(
            use_microphone=True, 
            record_duration=5, 
            play_audio=True
        )
    
    print("\nGoodbye!")