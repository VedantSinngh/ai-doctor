import os
import subprocess
import platform
from typing import Optional
from dotenv import load_dotenv
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TTSException(Exception):
    """Custom exception for TTS-related errors"""
    pass

class TTSEngine:
    """Base class for TTS engines"""
    def __init__(self):
        self.supported_formats = ['mp3', 'wav']
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> None:
        raise NotImplementedError
    
    def play_audio(self, file_path: str) -> None:
        """Play audio file using system's default player"""
        if not os.path.exists(file_path):
            raise TTSException(f"Audio file not found: {file_path}")
        
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                subprocess.run(['afplay', file_path], check=True)
            elif os_name == "Windows":  # Windows (use ffplay for compatibility)
                subprocess.run(['ffplay', '-autoexit', '-nodisp', file_path], check=True)
            elif os_name == "Linux":  # Linux
                try:
                    subprocess.run(['aplay', file_path], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        subprocess.run(['mpg123', file_path], check=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        subprocess.run(['ffplay', '-autoexit', '-nodisp', file_path], check=True)
            else:
                raise TTSException("Unsupported operating system")
        except subprocess.CalledProcessError as e:
            raise TTSException(f"Failed to play audio: {e}")
        except Exception as e:
            raise TTSException(f"Unexpected error playing audio: {e}")

class GTTSWrapper(TTSEngine):
    """Wrapper for Google Text-to-Speech (gTTS)"""
    def __init__(self, lang: str = 'en', slow: bool = False):
        super().__init__()
        self.lang = lang
        self.slow = slow
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> None:
        """Convert text to speech using gTTS"""
        try:
            lang = kwargs.get('lang', self.lang)
            slow = kwargs.get('slow', self.slow)
            
            file_ext = os.path.splitext(output_path)[1][1:].lower()
            if file_ext not in self.supported_formats:
                raise TTSException(f"Unsupported output format: {file_ext}")
            
            audio = gTTS(
                text=text,
                lang=lang,
                slow=slow
            )
            audio.save(output_path)
            logger.info(f"Audio saved successfully to {output_path}")
        except Exception as e:
            raise TTSException(f"gTTS synthesis failed: {e}")

class ElevenLabsWrapper(TTSEngine):
    """Wrapper for ElevenLabs Text-to-Speech"""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise TTSException("ElevenLabs API key not provided")
        
        self.client = ElevenLabs(api_key=self.api_key)
        self.default_voice = "Rachel"
        self.default_model = "eleven_turbo_v2"
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        speaker_boost: bool = True,
        **kwargs
    ) -> None:
        """Convert text to speech using ElevenLabs API"""
        try:
            file_ext = os.path.splitext(output_path)[1][1:].lower()
            if file_ext not in self.supported_formats:
                raise TTSException(f"Unsupported output format: {file_ext}")
            
            voice_obj = Voice(
                voice_id=voice or self.default_voice,
                settings=VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
                    speaker_boost=speaker_boost
                )
            )
            
            audio = self.client.generate(
                text=text,
                voice=voice_obj,
                model=model or self.default_model,
                output_format="mp3_22050_32" if file_ext == "mp3" else "pcm_16000"
            )
            
            elevenlabs.save(audio, output_path)
            logger.info(f"Audio saved successfully to {output_path}")
        except Exception as e:
            raise TTSException(f"ElevenLabs synthesis failed: {e}")

class TextToSpeechManager:
    """Manager class for handling multiple TTS engines"""
    def __init__(self):
        self.engines = {
            'gtts': GTTSWrapper(),
            'elevenlabs': ElevenLabsWrapper()
        }
    
    def get_engine(self, engine_name: str) -> TTSEngine:
        """Get a TTS engine by name"""
        engine = self.engines.get(engine_name.lower())
        if not engine:
            raise TTSException(f"Unknown TTS engine: {engine_name}")
        return engine
    
    def synthesize_and_play(
        self,
        text: str,
        output_path: str,
        engine_name: str = 'elevenlabs',
        play_audio: bool = True,
        **kwargs
    ) -> None:
        """Synthesize speech and optionally play it"""
        try:
            engine = self.get_engine(engine_name)
            engine.synthesize(text, output_path, **kwargs)
            
            if play_audio:
                engine.play_audio(output_path)
        except Exception as e:
            logger.error(f"Error in synthesize_and_play: {e}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        tts_manager = TextToSpeechManager()
        
        # Example with gTTS
        tts_manager.synthesize_and_play(
            text="Hello, this is an enhanced text-to-speech system using gTTS!",
            output_path="gtts_example.mp3",
            engine_name="elevenlabs",
            lang="en",
            slow=False
        )
        
        # Example with ElevenLabs (if API key is set)
        try:
            tts_manager.synthesize_and_play(
                text="Hello, this is an enhanced text-to-speech system using ElevenLabs!",
                output_path="elevenlabs_example.mp3",
                engine_name="elevenlabs",
                voice="Jessica",
                model="eleven_turbo_v2"
            )
        except TTSException as e:
            logger.warning(f"ElevenLabs example skipped: {e}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
