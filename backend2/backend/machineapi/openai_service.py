"""
OpenAI GPT-4o Vision Service for Video Analysis
Handles real-time video frame analysis with actual image understanding

This service provides true vision capabilities for drowsiness detection
by analyzing actual video frames with GPT-4o's multimodal model.
"""
import os
import base64
import json
import asyncio
from typing import Optional
from openai import OpenAI
from PIL import Image
import io


class OpenAIVideoAnalysisService:
    """Service for analyzing video frames using OpenAI GPT-4o Vision"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # GPT-4o is OpenAI's flagship multimodal model
        # Excellent for vision tasks with fast inference
        self.model = "gpt-4o"
        
        self.analysis_prompt = """
        Analyze this image for driver drowsiness detection. Look for:
        1. Eye closure/blinking patterns - are eyes open, half-closed, or closed?
        2. Head position and orientation - is head upright, tilted, or drooping?
        3. Facial expressions - signs of fatigue, stress, or alertness?
        4. Mouth - yawning, mouth open, or normal?
        5. Overall body language - posture, slouching, or attentive?

        Provide a brief analysis in JSON format with these exact fields:
        {
            "drowsiness_level": "awake" | "mildly drowsy" | "moderately drowsy" | "highly drowsy",
            "confidence": 0.0 to 1.0 (your confidence in this assessment),
            "observations": ["list", "of", "specific", "observations"],
            "recommended_action": "what should be done based on this analysis"
        }

        Criteria:
        - "awake": Eyes open, head upright, alert expression, good posture
        - "mildly drowsy": Slight eye closure, minor head nodding, some fatigue signs
        - "moderately drowsy": Frequent eye closure, head dropping, clear fatigue
        - "highly drowsy": Eyes mostly/fully closed, significant head drooping, urgent action needed
        """
    
    def encode_image(self, image: Image.Image, quality: int = 75) -> str:
        """Encode PIL image to base64 JPEG string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def preprocess_image(self, image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
        """Preprocess image for optimal analysis"""
        # Resize if too large (GPT-4o has context limits)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    async def analyze_frame(self, frame_data: bytes) -> dict:
        """Analyze a single video frame using GPT-4o Vision"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(frame_data))
            
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Encode to base64
            base64_image = self.encode_image(image)
            
            # Prepare the request with vision capabilities
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert driver drowsiness detection system. Analyze driver monitoring footage accurately and respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for consistent results
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Fallback parsing
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    result = json.loads(content[start:end])
                else:
                    result = {
                        "drowsiness_level": "unknown",
                        "confidence": 0.0,
                        "observations": ["Failed to parse analysis"],
                        "recommended_action": "Manual review required"
                    }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "drowsiness_level": "unknown",
                "confidence": 0.0,
                "observations": [f"Analysis failed: {str(e)}"],
                "recommended_action": "Check API connection and try again"
            }
    
    async def analyze_frame_sync(self, frame_data: bytes) -> dict:
        """Synchronous wrapper for frame analysis"""
        return await self.analyze_frame(frame_data)
    
    async def analyze_frame_batch(self, frames: list) -> list:
        """Process multiple frames in parallel"""
        tasks = [self.analyze_frame(frame) for frame in frames]
        return await asyncio.gather(*tasks)
    
    def estimate_cost(self, num_frames: int, image_size_kb: int = 100) -> dict:
        """Estimate API costs for a given number of frames"""
        # GPT-4o pricing (approximate, as of 2024)
        # Input: $5.00 / 1M tokens
        # Output: $15.00 / 1M tokens
        
        # Estimate tokens per frame (rough approximation)
        input_tokens_per_frame = 500  # ~100KB image + prompt
        output_tokens_per_frame = 100  # JSON response
        
        total_input_tokens = num_frames * input_tokens_per_frame
        total_output_tokens = num_frames * output_tokens_per_frame
        
        input_cost = (total_input_tokens / 1_000_000) * 5.00
        output_cost = (total_output_tokens / 1_000_000) * 15.00
        
        return {
            "frames": num_frames,
            "estimated_input_cost": f"${input_cost:.4f}",
            "estimated_output_cost": f"${output_cost:.4f}",
            "estimated_total_cost": f"${input_cost + output_cost:.4f}",
            "notes": "Actual costs may vary based on image size and response length"
        }


# Singleton instance
_openai_analysis_service: Optional[OpenAIVideoAnalysisService] = None


def get_openai_video_analysis_service() -> OpenAIVideoAnalysisService:
    """Get or create the OpenAI video analysis service singleton"""
    global _openai_analysis_service
    if _openai_analysis_service is None:
        _openai_analysis_service = OpenAIVideoAnalysisService()
    return _openai_analysis_service


async def process_frames_with_openai(frames: list) -> list:
    """Process multiple frames with OpenAI GPT-4o"""
    service = get_openai_video_analysis_service()
    return await service.analyze_frame_batch(frames)

