import os
import base64
from groq import Groq
from typing import Optional

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalysisException(Exception):
    """Custom exception for image analysis errors"""
    pass

def encode_image(image_path: str) -> Optional[str]:
    """
    Encode image to base64 string
    Args:
        image_path: Path to the image file
    Returns:
        Base64 encoded string or None if failed
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def analyze_image_with_query(
    query: str,
    encoded_image: str,
    model: str = "llama-3.2-90b-vision-preview",
    max_tokens: int = 512,
    temperature: float = 0.2
) -> str:
    """
    Analyze an image with a given query using Groq's vision model
    
    Args:
        query: The question/prompt about the image
        encoded_image: Base64 encoded image string
        model: The model to use for analysis
        max_tokens: Maximum tokens in response
        temperature: Creativity of response
        
    Returns:
        The analysis result as string
    """
    try:
        # Initialize Groq client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        if not client:
            raise ImageAnalysisException("Failed to initialize Groq client")
        
        # Prepare the message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]
        
        # Get the analysis from Groq API
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if not response.choices:
            raise ImageAnalysisException("No response from the model")
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return f"Sorry, I couldn't analyze the image. Error: {str(e)}"

# Example usage (for testing)
if __name__ == "__main__":
    # Test configuration
    test_image_path = "test_image.jpg"  # Replace with your test image
    test_query = "What's in this image? Is there anything medically concerning?"
    
    # Encode and analyze
    encoded_img = encode_image(test_image_path)
    if encoded_img:
        result = analyze_image_with_query(
            query=test_query,
            encoded_image=encoded_img,
            model="llama-3.2-90b-vision-preview"
        )
        print("Analysis Result:")
        print(result)
    else:
        print("Failed to encode test image")