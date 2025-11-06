# Golf Ball Path Tracker - 3 Putts

A real-time golf ball detection and path tracking system that uses WebSocket communication to track 3 putts and visualize the golf ball's path across the screen.

## Features

- ⛳ **Golf Ball Detection**: Uses YOLO model (`best.pt`) to detect golf balls in real-time
- 📊 **3-Putt Tracking**: Automatically tracks up to 3 putts per session
- 🎨 **Path Visualization**: Draws colorful paths for each putt (Green, Blue, Orange)
- 🔄 **Real-time Updates**: WebSocket-based communication for low-latency updates
- 📱 **Responsive UI**: Modern, mobile-friendly interface

## Files

- `golf_ball_detector.py` - FastAPI WebSocket server for golf ball detection
- `golf_ball.html` - Frontend HTML client with live camera feed
- `best.pt` - YOLO model file for golf ball detection

## Requirements

```bash
pip install fastapi uvicorn websockets opencv-python numpy torch ultralytics
```

## Usage

### 1. Start the Server

```bash
cd golfball
python golf_ball_detector.py
```

Or with uvicorn:
```bash
uvicorn golf_ball_detector:app --host 0.0.0.0 --port 8000
```

### 2. Open the HTML File

Open `golf_ball.html` in a web browser. You can:
- Serve it via a simple HTTP server: `python -m http.server 8080`
- Or open directly (may have CORS issues with WebSocket)

### 3. Use the Application

1. Click **"Use Camera"** to start your camera feed
2. Click **"Start Detection"** to begin tracking
3. Make your putts - the system will automatically:
   - Detect the golf ball
   - Track its path in real-time
   - Complete each putt when the ball stops moving
   - Move to the next putt automatically
4. View results in the right panel showing all 3 putts

## Configuration

### WebSocket URL

By default, the HTML connects to `ws://localhost:8000`. To change this, add a query parameter:
```
golf_ball.html?ws=ws://your-server:8000
```

### Detection Settings

In `golf_ball_detector.py`, you can adjust:
- `TARGET_SIZE = 640` - Image processing size
- `CONFIDENCE_THRESHOLD = 0.25` - Detection confidence threshold
- `STILLNESS_THRESHOLD = 0.5` - Seconds of stillness to complete putt
- `MIN_PUTT_POINTS = 5` - Minimum points for valid putt
- `MAX_PUTT_DURATION = 10.0` - Maximum seconds per putt

## How It Works

1. **Camera Feed**: Browser captures video frames from camera
2. **Frame Processing**: Frames are sent to server via WebSocket as binary data
3. **Detection**: Server uses YOLO model to detect golf ball position
4. **Trajectory Tracking**: Server tracks ball movement and builds path
5. **Putt Completion**: When ball stops moving (stillness threshold), putt is finalized
6. **Path Drawing**: Frontend draws paths on canvas overlay in real-time

## Path Colors

- **Putt 1**: Green (#22c55e)
- **Putt 2**: Blue (#3b82f6)
- **Putt 3**: Orange (#f59e0b)

## Troubleshooting

- **Model not loading**: Ensure `best.pt` is in the same directory as `golf_ball_detector.py`
- **WebSocket connection failed**: Check server is running and port 8000 is accessible
- **Camera not working**: Grant camera permissions in browser
- **No detections**: Adjust `CONFIDENCE_THRESHOLD` or check lighting conditions

## License

This project is part of the VAST AI Putter Detection system.

