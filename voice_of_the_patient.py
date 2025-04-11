import os
import logging
from typing import Optional, Tuple
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq
import sounddevice as sd  # For additional audio functionality
import numpy as np  # For audio processing
import wave  # For WAV file handling

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioException(Exception):
    """Custom exception for audio-related errors"""
    pass

class AudioRecorder:
    """Class for handling audio recording operations"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.sample_rate = 44100  # Standard sample rate
        self.channels = 1  # Mono audio
    
    def _validate_output_path(self, file_path: str) -> None:
        """Validate the output file path"""
        if not file_path.lower().endswith(('.mp3', '.wav')):
            raise AudioException("Output file must be MP3 or WAV format")
        
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def record_audio(
        self,
        file_path: str,
        timeout: int = 20,
        phrase_time_limit: Optional[int] = None,
        ambient_adjust_duration: float = 1.0,
        bitrate: str = "128k"
    ) -> Tuple[bool, str]:
        """
        Record audio from microphone and save to file.
        
        Args:
            file_path: Output file path (MP3 or WAV)
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for recording
            ambient_adjust_duration: Seconds for noise adjustment
            bitrate: Quality of MP3 output
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            self._validate_output_path(file_path)
            
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(
                    source, 
                    duration=ambient_adjust_duration
                )
                
                logger.info("Start speaking now...")
                audio_data = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                logger.info("Recording complete.")
                
                # Convert and save audio
                wav_data = audio_data.get_wav_data()
                audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                
                if file_path.lower().endswith('.mp3'):
                    audio_segment.export(
                        file_path, 
                        format="mp3", 
                        bitrate=bitrate
                    )
                else:
                    audio_segment.export(
                        file_path, 
                        format="wav"
                    )
                
                logger.info(f"Audio successfully saved to {file_path}")
                return True, "Recording successful"
                
        except sr.WaitTimeoutError:
            msg = "No speech detected before timeout"
            logger.warning(msg)
            return False, msg
        except sr.UnknownValueError:
            msg = "Could not understand audio"
            logger.warning(msg)
            return False, msg
        except Exception as e:
            msg = f"Recording failed: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def record_audio_with_sounddevice(
        self,
        file_path: str,
        duration: float = 5.0,
        sample_rate: int = 44100
    ) -> None:
        """
        Alternative recording method using sounddevice library
        
        Args:
            file_path: Output WAV file path
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
        """
        try:
            self._validate_output_path(file_path)
            if not file_path.lower().endswith('.wav'):
                raise AudioException("sounddevice recording only supports WAV format")
            
            logger.info(f"Recording for {duration} seconds...")
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()  # Wait until recording is finished
            
            # Save as WAV file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(sample_rate)
                wf.writeframes(recording.tobytes())
            
            logger.info(f"Recording saved to {file_path}")
            
        except Exception as e:
            logger.error(f"sounddevice recording failed: {e}")
            raise AudioException(f"Recording failed: {e}")

class SpeechToText:
    """Class for handling speech-to-text transcription"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise AudioException("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.default_model = "whisper-large-v3"
    
    def transcribe_audio(
        self,
        audio_file_path: str,
        model: Optional[str] = None,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        word_timestamps: bool = False
    ) -> Tuple[bool, str]:
        """
        Transcribe audio file to text using Groq API
        
        Args:
            audio_file_path: Path to audio file
            model: STT model to use
            language: Language of audio
            prompt: Optional context prompt
            temperature: Creativity of transcription
            word_timestamps: Include word timestamps
            
        Returns:
            Tuple of (success: bool, transcription: str)
        """
        try:
            if not os.path.exists(audio_file_path):
                raise AudioException(f"Audio file not found: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=model or self.default_model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    response_format="json",
                    timestamp_granularities=["word"] if word_timestamps else None
                )
            
            # Handle different response formats
            if hasattr(transcription, 'text'):
                return True, transcription.text
            elif isinstance(transcription, dict) and 'text' in transcription:
                return True, transcription['text']
            else:
                raise AudioException("Unexpected transcription response format")
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return False, f"Transcription error: {str(e)}"

class AudioProcessor:
    """Main class for audio processing pipeline"""
    
    def __init__(self):
        self.recorder = AudioRecorder()
        self.transcriber = SpeechToText()
    
    def record_and_transcribe(
        self,
        output_path: str = "recording.mp3",
        record_timeout: int = 20,
        phrase_limit: Optional[int] = None,
        model: str = "whisper-large-v3",
        language: str = "en"
    ) -> Tuple[bool, str]:
        """
        Complete pipeline: record audio and transcribe it
        
        Returns:
            Tuple of (success: bool, result: str)
        """
        try:
            # Record audio
            success, msg = self.recorder.record_audio(
                output_path,
                timeout=record_timeout,
                phrase_time_limit=phrase_limit
            )
            if not success:
                return False, msg
            
            # Transcribe audio
            success, transcription = self.transcriber.transcribe_audio(
                output_path,
                model=model,
                language=language
            )
            
            return success, transcription
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False, f"Processing error: {str(e)}"

# Example usage
if __name__ == "__main__":
    try:
        processor = AudioProcessor()
        
        # Example recording and transcription
        audio_file = "patient_recording.mp3"
        logger.info("Starting recording...")
        
        success, result = processor.record_and_transcribe(
            output_path=audio_file,
            record_timeout=15,
            phrase_limit=10
        )
        
        if success:
            logger.info("Transcription successful!")
            logger.info(f"Result: {result}")
        else:
            logger.error(f"Error: {result}")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")