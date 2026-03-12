"""
WebSocket Consumer for Real-time Video Analysis
Handles video frame streaming from Flutter app and returns AI analysis
"""
import json
import asyncio
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .openai_service import get_openai_video_analysis_service


class VideoAnalysisConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time video drowsiness analysis
    
    Client sends: Base64 encoded video frames
    Server returns: AI analysis results with drowsiness detection
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_service = None
        self.frame_count = 0
        self.analysis_interval = 1  # Analyze every Nth frame
        self.last_analysis_result = None
    
    async def connect(self):
        """Handle WebSocket connection"""
        self.group_name = "video_analysis"
        
        # Accept the connection
        await self.accept()
        
        # Initialize the OpenAI GPT analysis service
        try:
            self.analysis_service = get_openai_video_analysis_service()
            await self.send(text_data=json.dumps({
                "type": "connection_established",
                "message": "Connected to video analysis service",
                "status": "ready"
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                "type": "connection_error",
                "message": f"Failed to initialize analysis service: {str(e)}",
                "status": "error"
            }))
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        print(f"WebSocket disconnected with code: {close_code}")
        # Clean up if needed
    
    async def receive(self, text_data=None, bytes_data=None):
        """Handle incoming video frames from Flutter app"""
        try:
            if text_data:
                # Text message - could be configuration or control messages
                data = json.loads(text_data)
                
                if data.get("type") == "configure":
                    # Configure analysis parameters
                    self.analysis_interval = data.get("interval", 1)
                    await self.send(text_data=json.dumps({
                        "type": "configuration_acknowledged",
                        "interval": self.analysis_interval
                    }))
                    return
            
            if bytes_data:
                # Binary data - video frame
                self.frame_count += 1
                
                # Only analyze every Nth frame to reduce load
                if self.frame_count % self.analysis_interval == 0:
                    await self.process_frame(bytes_data)
                else:
                    # Send acknowledgment for skipped frames
                    await self.send(text_data=json.dumps({
                        "type": "frame_received",
                        "frame_number": self.frame_count,
                        "analyzed": False
                    }))
                    
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": "Invalid JSON format"
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def process_frame(self, frame_data: bytes):
        """Process video frame with OpenAI GPT-4o"""
        try:
            # Send processing status
            await self.send(text_data=json.dumps({
                "type": "processing",
                "frame_number": self.frame_count,
                "message": "Analyzing frame..."
            }))
            
            # Analyze frame using OpenAI GPT-4o
            if self.analysis_service:
                result = await self.analysis_service.analyze_frame(frame_data)
                self.last_analysis_result = result
                
                # Send analysis result back to client
                await self.send(text_data=json.dumps({
                    "type": "analysis_result",
                    "frame_number": self.frame_count,
                    "data": result
                }))
            else:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": "Analysis service not initialized"
                }))
                
        except Exception as e:
            await self.send(text_data=json.dumps({
                "type": "analysis_error",
                "frame_number": self.frame_count,
                "error": str(e)
            }))
    
    async def send_analysis_result(self, result: dict):
        """Send analysis result to WebSocket client"""
        await self.send(text_data=json.dumps({
            "type": "analysis_result",
            "data": result
        }))

