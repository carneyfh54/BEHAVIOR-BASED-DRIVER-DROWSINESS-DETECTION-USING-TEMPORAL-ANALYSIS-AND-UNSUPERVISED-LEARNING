#!/bin/bash
echo "🚀 Starting AWAKE Drowsiness Detection System..."

# Start Daphne in background
echo "🔧 Starting Django backend..."
source venv/bin/activate
python -m daphne -b 0.0.0.0 -p 8000 drowsiness_detector.asgi:application &
DAPHNE_PID=$!
echo "✅ Backend running (PID $DAPHNE_PID)"

# Wait for Daphne to be ready
sleep 2

# Start Flutter
echo "📱 Starting Flutter frontend..."
flutter run -d chrome

# When Flutter exits, kill Daphne
echo "🛑 Shutting down backend..."
kill $DAPHNE_PID
