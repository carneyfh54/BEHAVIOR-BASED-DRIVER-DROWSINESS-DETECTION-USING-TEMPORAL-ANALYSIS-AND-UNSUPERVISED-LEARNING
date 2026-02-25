# Driver Drowsiness Detection - Real-time Video Analysis

This project implements real-time driver drowsiness detection using WebSocket video streaming and Groq AI for analysis.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Flutter App   │────▶│   WebSocket      │────▶│  Django Backend │
│  (Video Input)  │     │   (ws://...)     │     │  (AI Analysis)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                                               │
        │                                               │
        │              ┌─────────────────┐              │
        └─────────────▶│   Groq API      │◀─────────────┘
                       │   (Vision AI)   │
                       └─────────────────┘
```

## Prerequisites

### Backend Requirements
- Python 3.9+
- Django 4.2+
- Groq API Key (free tier available)

### Flutter App Requirements
- Flutter 3.10+
- Camera permissions

## Backend Setup

### 1. Navigate to backend directory
```bash
cd backend2
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 5. Get Groq API Key
1. Visit https://console.groq.com/
2. Create a free account
3. Generate an API key
4. Add it to your `.env` file

### 6. Run migrations
```bash
cd backend
python manage.py migrate
```

### 7. Start the server
```bash
python manage.py daphne backend.asgi:application
```

The server will start at `http://localhost:8000`

## Flutter App Setup

### 1. Update dependencies
```bash
flutter pub get
```

### 2. Configure WebSocket URL
In your Flutter code, set the server URL:
```dart
// Example connection
await VideoAnalysisService().connect('localhost', port: 8000);
```

For physical devices, use your computer's IP address:
```dart
await VideoAnalysisService().connect('192.168.1.x', port: 8000);
```

## API Endpoints

### HTTP Endpoints
- `GET /api/health/` - Health check
- `GET /api/ws-info/` - WebSocket information

### WebSocket Endpoint
- `ws://<server>/ws/video-analysis/` - Real-time video analysis

## WebSocket Message Protocol

### Client → Server

**Configuration Message:**
```json
{
  "type": "configure",
  "interval": 1
}
```

**Video Frame:**
- Send binary JPEG data
- Recommended: 1-2 frames per second

### Server → Client

**Connection Established:**
```json
{
  "type": "connection_established",
  "message": "Connected to video analysis service",
  "status": "ready"
}
```

**Analysis Result:**
```json
{
  "type": "analysis_result",
  "frame_number": 5,
  "data": {
    "drowsiness_level": "awake",
    "confidence": 0.95,
    "observations": ["Eyes open", "Head upright"],
    "recommended_action": "Continue monitoring"
  }
}
```

**Processing Status:**
```json
{
  "type": "processing",
  "frame_number": 5,
  "message": "Analyzing frame..."
}
```

## Drowsiness Levels

| Level | Description | Color |
|-------|-------------|-------|
| `awake` | Driver is alert | 🟢 Green |
| `mildly drowsy` | Slight fatigue signs | 🟡 Yellow |
| `moderately drowsy` | Clear drowsiness signs | 🟠 Orange |
| `highly drowsy` | Severe drowsiness, immediate action needed | 🔴 Red |

## Testing

### Backend Tests
```bash
cd backend2/backend
python manage.py test
```

### Manual WebSocket Test
```python
# Using websockets library
import asyncio
import base64

async def test_websocket():
    async with websockets.connect('ws://localhost:8000/ws/video-analysis/') as ws:
        # Wait for connection
        response = await ws.recv()
        print(response)
        
        # Send a test frame (example)
        # with open('test_frame.jpg', 'rb') as f:
        #     await ws.send(f.read())
        
        # Get analysis result
        result = await ws.recv()
        print(result)

asyncio.run(test_websocket())
```

## Performance Optimization

### For Real-time Analysis
1. **Frame Rate**: Send 1-2 frames per second
2. **Image Quality**: Use JPEG quality 70-80%
3. **Resolution**: 640x480 or 720p is sufficient
4. **Interval**: Set analysis interval to 1-2

### Backend Scaling
- Use Redis channel layer for production
- Deploy with multiple Daphne workers
- Consider GPU acceleration for image processing

## Troubleshooting

### WebSocket Connection Failed
1. Check if server is running: `curl http://localhost:8000/api/health/`
2. Verify firewall allows port 8000
3. Check ALLOWED_HOSTS in settings

### No Analysis Results
1. Verify Groq API key is set correctly
2. Check backend logs for errors
3. Ensure frames are valid JPEG images

### High Latency
1. Reduce frame rate (send fewer frames)
2. Lower image resolution
3. Increase analysis interval

## Security Considerations

- **API Keys**: Never commit `.env` files
- **Production**: Use HTTPS/WSS for WebSocket
- **Validation**: Validate all incoming frames
- **Rate Limiting**: Implement request throttling

## License

MIT License

