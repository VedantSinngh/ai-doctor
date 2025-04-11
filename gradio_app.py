import os
import base64
import logging
import tempfile
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Third-party imports
from groq import Groq
from dotenv import load_dotenv
import gradio as gr
from gtts import gTTS
from elevenlabs import ElevenLabs, Voice, VoiceSettings
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Healthcare system prompt - refined for professional medical consultation
HEALTHCARE_SYSTEM_PROMPT = """
You are a healthcare professional providing a preliminary assessment. 
Based on the visual information provided, offer your observations about potential medical concerns.
If you identify possible conditions, suggest appropriate next steps or remedies, while emphasizing 
the importance of seeking professional medical advice for proper diagnosis.
Present your assessment in a clear, compassionate manner using professional medical terminology 
while remaining accessible. Address the patient directly in a reassuring tone.
Keep your assessment concise (approximately 2-3 sentences) and focused on observable symptoms.
Begin your response immediately with your assessment.
"""

class ImageProcessor:
    """Handles image encoding and medical image analysis using AI."""
    
    def __init__(self, api_key: str, model: str = "llama-3.2-11b-vision-preview"):
        """
        Initialize the image processor.
        
        Args:
            api_key: The API key for the Groq service
            model: The AI model to use for image analysis
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        
    @staticmethod
    def encode_image(image_path: str) -> Optional[str]:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image or None if encoding fails
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return None
    
    def analyze_image(self, query: str, encoded_image: str) -> str:
        """
        Analyze an image with the specified query using a vision AI model.
        
        Args:
            query: The prompt for the AI model
            encoded_image: Base64 encoded image
            
        Returns:
            The AI's analysis response
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                }
            ]
            response = self.client.chat.completions.create(
                messages=messages, 
                model=self.model, 
                max_tokens=512, 
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return "I'm unable to provide an assessment at this time due to a technical issue. Please try again later."


class AudioProcessor:
    """Handles audio recording, processing, and transcription."""
    
    def __init__(self, api_key: str):
        """
        Initialize the audio processor.
        
        Args:
            api_key: The API key for the Groq service
        """
        self.recognizer = sr.Recognizer()
        self.client = Groq(api_key=api_key)
        
    def record_audio(self, file_path: str, timeout: int = 15, phrase_time_limit: int = 10) -> bool:
        """
        Record audio from the microphone.
        
        Args:
            file_path: Path to save the recorded audio
            timeout: Maximum time to wait for speech
            phrase_time_limit: Maximum time for a single phrase
            
        Returns:
            True if recording was successful, False otherwise
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                logger.info("Please describe your symptoms...")
                audio_data = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                wav_data = audio_data.get_wav_data()
                audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                audio_segment.export(file_path, format="mp3", bitrate="128k")
                logger.info(f"Audio recorded successfully")
                return True
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return False
            
    def transcribe_audio(self, audio_filepath: str) -> Tuple[bool, str]:
        """
        Transcribe audio using speech recognition.
        
        Args:
            audio_filepath: Path to the audio file
            
        Returns:
            Tuple of (success, transcription text)
        """
        try:
            if not os.path.exists(audio_filepath):
                return False, "Audio file not found"
                
            with open(audio_filepath, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    language="en",
                    temperature=0.0
                )
            return True, transcription.text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return False, "I couldn't understand the audio. Please try speaking clearly or typing your symptoms."


class TextToSpeechManager:
    """Manages text-to-speech conversion using multiple engines."""
    
    def __init__(self, elevenlabs_api_key: str):
        """
        Initialize the text-to-speech manager.
        
        Args:
            elevenlabs_api_key: API key for ElevenLabs
        """
        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        # Professional healthcare voice - update with your preferred voice ID
        self.default_voice_id = "cgSgspJ2msm6clMCkdW9"  
        
    def synthesize(self, text: str, output_path: str, engine: str = "elevenlabs") -> bool:
        """
        Synthesize text to speech.
        
        Args:
            text: The text to convert to speech
            output_path: Path to save the audio file
            engine: The TTS engine to use ("elevenlabs" or "gtts")
            
        Returns:
            True if synthesis was successful, False otherwise
        """
        try:
            if engine == "elevenlabs":
                voice = Voice(
                    voice_id=self.default_voice_id,
                    settings=VoiceSettings(
                        stability=0.6,  # More stability for medical context
                        similarity_boost=0.7, 
                        style=0.0, 
                        speaker_boost=True
                    )
                )
                audio = self.elevenlabs_client.generate(
                    text=text,
                    voice=voice,
                    model="eleven_turbo_v2",
                    output_format="mp3_22050_32"
                )
                with open(output_path, "wb") as f:
                    f.write(audio)
            else:  # Fallback to gTTS
                audio = gTTS(text=text, lang="en", slow=False)
                audio.save(output_path)
            logger.info(f"Speech synthesis completed successfully")
            return True
        except Exception as e:
            logger.error(f"TTS synthesis failed with {engine}: {e}")
            # Try alternative if primary engine fails
            if engine == "elevenlabs":
                logger.info("Attempting fallback to gTTS")
                return self.synthesize(text, output_path, "gtts")
            return False


class MedicalConsultationApp:
    """Main application class for the medical consultation system."""
    
    def __init__(self):
        """Initialize the medical consultation application."""
        # Check for required API keys
        required_keys = ["GROQ_API_KEY", "ELEVENLABS_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}. Please check your .env file.")
            
        # Initialize components
        groq_api_key = os.getenv("GROQ_API_KEY")
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        self.image_processor = ImageProcessor(api_key=groq_api_key)
        self.audio_processor = AudioProcessor(api_key=groq_api_key)
        self.tts_manager = TextToSpeechManager(elevenlabs_api_key=elevenlabs_api_key)
        
        # Create temp directory for session files
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Temporary directory created at {self.temp_dir}")
        
    def process_consultation(
        self, 
        audio_filepath: Optional[str], 
        image_filepath: Optional[str]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Process a medical consultation request.
        
        Args:
            audio_filepath: Path to the audio file containing patient symptoms
            image_filepath: Path to the image file showing the medical condition
            
        Returns:
            Tuple of (transcribed_text, healthcare_response, response_audio_path)
        """
        # Output file path
        output_audio_path = str(self.temp_dir / "healthcare_response.mp3")
        
        # Process audio input
        if audio_filepath:
            success, transcribed_text = self.audio_processor.transcribe_audio(audio_filepath)
            if not success:
                transcribed_text = "Unable to transcribe audio"
        else:
            transcribed_text = "No audio description provided"
            
        # Process image input
        if image_filepath:
            encoded_image = self.image_processor.encode_image(image_filepath)
            if encoded_image:
                query = f"{HEALTHCARE_SYSTEM_PROMPT} Patient description: {transcribed_text}"
                healthcare_response = self.image_processor.analyze_image(query, encoded_image)
            else:
                healthcare_response = "I'm unable to process the provided image. Please try uploading a clearer image."
        else:
            healthcare_response = "No image was provided for assessment. Please upload an image of the affected area."
            
        # Generate audio response
        if healthcare_response:
            success = self.tts_manager.synthesize(healthcare_response, output_audio_path)
            if not success:
                output_audio_path = None
                
        return transcribed_text, healthcare_response, output_audio_path
        
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface for the application.
        
        Returns:
            Configured Gradio interface
        """
        # Define a medical theme with blue tones
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="blue",
            neutral_hue="slate"
        )
        
        with gr.Blocks(theme=theme) as interface:
            # CSS for custom styling
            gr.Markdown("""
            <style>
            .container {
                max-width: 1000px;
                margin: 0 auto;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .header {
                text-align: center;
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 0.5rem;
                background: linear-gradient(90deg, #e0f2fe, #bfdbfe);
            }
            
            .header h1 {
                color: #1e40af;
                margin: 0;
                font-size: 2rem;
            }
            
            .header p {
                color: #475569;
                margin: 0.5rem 0 0 0;
            }
            
            .input-container, .output-container {
                border: 1px solid #e2e8f0;
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            
            .instruction-box {
                background-color: #f0f9ff;
                border-left: 4px solid #3b82f6;
                padding: 1rem;
                border-radius: 0.25rem;
                margin-bottom: 1.5rem;
            }
            
            .disclaimer {
                background-color: #fff1f2;
                border-left: 4px solid #e11d48;
                padding: 1rem;
                border-radius: 0.25rem;
                margin-top: 1.5rem;
                font-size: 0.9rem;
            }
            
            .step-header {
                color: #1e40af;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 0.75rem;
                margin-top: 0;
                margin-bottom: 1.25rem;
                font-weight: 600;
                font-size: 1.25rem;
            }
            
            .footer {
                text-align: center;
                color: #64748b;
                font-size: 0.8rem;
                margin-top: 2rem;
            }
            </style>
            """)
            
            # Header section
            gr.HTML("""
            <div class="container">
                <div class="header">
                    <h1>MediScan: AI-Powered Healthcare Assessment</h1>
                    <p>Get a preliminary assessment of skin conditions and symptoms through advanced AI analysis</p>
                </div>
            </div>
            """)
            
            with gr.Row(equal_height=True):
                # Left column - Input section
                with gr.Column():
                    gr.HTML("""
                    <div class="input-container">
                        <h2 class="step-header">Step 1: Provide Information</h2>
                        <div class="instruction-box">
                            <h3 style="margin-top: 0; color: #1e40af;">How to Use This Tool</h3>
                            <ol>
                                <li>Upload a clear image of the affected area</li>
                                <li>Optionally record audio describing your symptoms</li>
                                <li>Click "Get Assessment" for AI analysis</li>
                                <li>Review your assessment results</li>
                            </ol>
                        </div>
                    """)
                    
                    # Image input
                    image_input = gr.Image(
                        type="filepath",
                        label="Upload Image of Affected Area",
                        elem_id="image-input"
                    )
                    
                    # Audio input
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Describe Your Symptoms (Optional)",
                        elem_id="audio-input"
                    )
                    
                    # Submit button
                    submit_btn = gr.Button(
                        "Get Assessment", 
                        variant="primary",
                        elem_id="submit-btn",
                        size="lg"
                    )
                    
                    gr.HTML("""
                        <div class="disclaimer">
                            <strong>Important Notice:</strong> This tool provides a preliminary assessment only 
                            and is not a substitute for professional medical care. Always consult with a qualified 
                            healthcare provider for proper diagnosis and treatment.
                        </div>
                    </div>
                    """)
                
                # Right column - Output section
                with gr.Column():
                    gr.HTML("""
                    <div class="output-container">
                        <h2 class="step-header">Step 2: Review Your Assessment</h2>
                    """)
                    
                    # Transcription output
                    transcription_output = gr.Textbox(
                        label="Your Described Symptoms",
                        elem_id="transcription-output",
                        lines=2
                    )
                    
                    # Assessment output
                    assessment_output = gr.Textbox(
                        label="Professional Assessment",
                        elem_id="assessment-output",
                        lines=6
                    )
                    
                    # Audio output
                    audio_output = gr.Audio(
                        label="Audio Assessment",
                        type="filepath",
                        elem_id="audio-output"
                    )
                    
                    gr.HTML("""
                    </div>
                    """)
            
            # Footer
            gr.HTML("""
            <div class="container">
                <div class="footer">
                    <p>© 2025 MediScan - AI-Powered Healthcare Assessment Tool</p>
                    <p>This application uses AI technologies from Groq and ElevenLabs</p>
                </div>
            </div>
            """)
            
            # Set up event handlers
            submit_btn.click(
                fn=self.process_consultation,
                inputs=[audio_input, image_input],
                outputs=[transcription_output, assessment_output, audio_output],
                api_name="process_consultation"
            )
            
        return interface


if __name__ == "__main__":
    try:
        app = MedicalConsultationApp()
        interface = app.create_interface()
        interface.launch(debug=False)
    except ValueError as e:
        logger.critical(f"Application initialization failed: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")