from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json


def health_check(request):
    """Health check endpoint for the API"""
    return JsonResponse({
        "status": "healthy",
        "service": "driver-drowsiness-detection",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "ws://<server>/ws/video-analysis/",
            "health": "/api/health/"
        }
    })


@require_http_methods(["GET"])
def websocket_info(request):
    """Return WebSocket connection information"""
    return JsonResponse({
        "websocket_endpoint": "ws://<server>/ws/video-analysis/",
        "protocol": "websocket",
        "features": [
            "real-time video frame analysis",
            "drowsiness detection",
            "configurable analysis interval",
            "JSON response format"
        ],
        "message_format": {
            "send": {
                "binary": "JPEG encoded video frame bytes",
                "text": "JSON configuration messages"
            },
            "receive": {
                "analysis_result": {
                    "type": "analysis_result",
                    "data": {
                        "drowsiness_level": "string",
                        "confidence": "float",
                        "observations": "list",
                        "recommended_action": "string"
                    }
                }
            }
        }
    })


# Create your views here.
