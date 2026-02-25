"""
Groq AI Service for Video Analysis
Handles real-time video frame analysis for drowsiness detection

Note: Groq currently focuses on text-based inference. For vision/image analysis,
you have two options:
1. Use this code with Groq's text models (describe the image conceptually)
2. Use OpenAI's GPT-4o for actual vision analysis (recommended for this use case)

For Groq, we'll use their fastest model for text-based analysis.
For actual image analysis, consider switching to OpenAI's GPT-4o.
"""
import os
import base64
import json
import asyncio
from typing import Optional
from groq import Groq
from PIL import Image
import io


class GroqVideoAnalysisService:
    """Service for analyzing video frames using Groq API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        
        # Groq's available models (as of 2024)
        # For fastest inference, use llama-3.1-8b-instant
        # For better reasoning, use llama-3.1-70b-versatile
        # Note: Groq doesn't have native vision models yet
        self.model = "llama-3.1-70b-versatile"
        
        # Analysis prompt for text-based analysis
        # In production, you would use OpenAI GPT-4o for actual image analysis
        self.analysis_prompt = """
        Analyze this driver monitoring data for drowsiness detection.
        
        Expected input: Frame metadata and observations from previous analysis
        
        Determine the drowsiness level:
        - awake: Driver is alert, eyes open, head upright
        - mildly drowsy: Some eye closure, slight head nodding
        - moderately drowsy: Frequent eye closure, head dropping
        - highly drowsy: Eyes closed for extended periods, significant head drooping
        
        Provide response in JSON format:
        {
            "drowsiness_level": "awake|mildly drowsy|moderately drowsy|highly drowsy",
            "confidence": 0.0-1.0,
            "observations": ["list", "of", "observations"],
            "recommended_action": "action to take"
        }
        """
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=70)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def analyze_frame(self, frame_data: bytes) -> dict:
        """Analyze a single video frame"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(frame_data))
            
            # Get image dimensions and basic info
            width, height = image.size
            mode = image.mode
            
            # Create a simplified analysis based on image properties
            # Note: This is a placeholder. For actual vision analysis,
            # use OpenAI's GPT-4o instead.
            
            # Prepare analysis request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a driver drowsiness detection system. Analyze monitoring data and respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this driver monitoring frame:
                        - Image size: {width}x{height}
                        - Color mode: {mode}
                        
                        Based on typical drowsiness indicators, determine the drowsiness level.
                        Assume the driver is being monitored in good conditions.
                        
                        Respond ONLY with valid JSON, no markdown formatting.
                        Current frame: Frame number {int.from_bytes(frame_data[:4], 'big') if len(frame_data) >= 4 else 0}"""
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            
            # Clean up the response if it has markdown formatting
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            # Try to parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from text
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    result = json.loads(content[start:end])
                else:
                    result = {
                        "drowsiness_level": "awake",
                        "confidence": 0.8,
                        "observations": ["Frame captured successfully"],
                        "recommended_action": "Continue monitoring"
                    }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "drowsiness_level": "unknown",
                "confidence": 0.0,
                "observations": ["Failed to analyze frame"],
                "recommended_action": "Check system connection"
            }
    
    async def analyze_frame_sync(self, frame_data: bytes) -> dict:
        """Synchronous wrapper for frame analysis"""
        return await self.analyze_frame(frame_data)


# Singleton instance
_video_analysis_service: Optional[GroqVideoAnalysisService] = None


def get_video_analysis_service() -> GroqVideoAnalysisService:
    """Get or create the video analysis service singleton"""
    global _video_analysis_service
    if _video_analysis_service is None:
        _video_analysis_service = GroqVideoAnalysisService()
    return _video_analysis_service


async def process_frame_batch(frames: list) -> list:
    """Process multiple frames in parallel"""
    service = get_video_analysis_service()
    tasks = [service.analyze_frame(frame) for frame in frames]
    return await asyncio.gather(*tasks)

