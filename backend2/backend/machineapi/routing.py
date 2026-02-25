"""
WebSocket URL Configuration for Video Analysis
"""
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Video analysis WebSocket endpoint
    re_path(r'ws/video-analysis/$', consumers.VideoAnalysisConsumer.as_asgi()),
    
    # Alternative endpoint name
    re_path(r'ws/video/$', consumers.VideoAnalysisConsumer.as_asgi()),
]

