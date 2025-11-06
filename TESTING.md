# Testing Guide for Golf Ball Path Tracker

## Quick Start Testing

### Step 1: Install Dependencies

Open a terminal in the `golfball` folder and run:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn websockets opencv-python numpy torch ultralytics
```

### Step 2: Start the Server

In the `golfball` folder, run:

```bash
python golf_ball_detector.py
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Model loaded on cpu (or cuda)
INFO:     Model warmed up
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Open the HTML File

**Option A: Direct File Open**
- Simply double-click `golf_ball.html` to open in your browser
- Note: Some browsers may block WebSocket connections from `file://` protocol

**Option B: Using a Local Server (Recommended)**

1. Open a new terminal in the `golfball` folder
2. Start a simple HTTP server:
   ```bash
   # Python 3
   python -m http.server 8080
   
   # Or using Node.js (if installed)
   npx http-server -p 8080
   ```
3. Open browser and go to: `http://localhost:8080/golf_ball.html`

### Step 4: Test the Application

1. **Start Camera**
   - Click "Use Camera" button
   - Grant camera permissions when prompted
   - You should see your camera feed

2. **Start Detection**
   - Click "Start Detection" button
   - The status should change to "Detecting golf ball — Make your first putt"

3. **Test Detection**
   - Point camera at a golf ball
   - The system should detect the ball and start tracking
   - Move the ball - you should see a path being drawn in real-time

4. **Complete a Putt**
   - Move the golf ball across the screen
   - Stop moving the ball for ~0.5 seconds
   - The system should automatically complete Putt 1
   - Repeat for Putts 2 and 3

## Testing Checklist

- [ ] Server starts without errors
- [ ] Model loads successfully (check console output)
- [ ] HTML page opens in browser
- [ ] Camera permission granted
- [ ] Camera feed displays
- [ ] WebSocket connection established (check browser console)
- [ ] Golf ball detected when visible
- [ ] Path draws in real-time as ball moves
- [ ] Putt completes automatically when ball stops
- [ ] All 3 putts can be tracked
- [ ] Results panel shows completed putts

## Troubleshooting

### Server Issues

**Problem: "Module not found" errors**
- Solution: Install dependencies with `pip install -r requirements.txt`

**Problem: "Failed to load model"**
- Solution: Ensure `best.pt` is in the `golfball` folder or in `weights/best.pt`
- Check console output for the exact error

**Problem: Port 8000 already in use**
- Solution: Change port in `golf_ball_detector.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
  ```
- Update HTML WebSocket URL accordingly

### Browser Issues

**Problem: WebSocket connection fails**
- Solution: Use a local HTTP server instead of opening file directly
- Check browser console for CORS or connection errors

**Problem: Camera not working**
- Solution: Check browser permissions
- Try a different browser (Chrome, Firefox, Edge)
- Ensure camera is not being used by another application

**Problem: No detections**
- Solution: 
  - Check lighting conditions
  - Ensure golf ball is clearly visible
  - Adjust `CONFIDENCE_THRESHOLD` in code (lower = more sensitive)
  - Check console for detection errors

### Detection Issues

**Problem: Ball not detected**
- Check server console for errors
- Try lowering confidence threshold in `golf_ball_detector.py`:
  ```python
  CONFIDENCE_THRESHOLD = 0.15  # Lower value = more sensitive
  ```

**Problem: Path not drawing**
- Check browser console for JavaScript errors
- Verify WebSocket messages are being received
- Check that `path_points` are in the response

**Problem: Putt not completing**
- Adjust stillness threshold in code:
  ```python
  self.STILLNESS_THRESHOLD = 0.3  # Lower = completes faster
  ```

## Manual Testing with Test Images

If you want to test without a camera:

1. Modify the code to accept image uploads
2. Use test images with golf balls
3. Send frames via WebSocket manually

## Performance Testing

Monitor:
- **FPS**: Should be ~15 FPS (66ms per frame)
- **Latency**: Detection should be <100ms
- **Memory**: Check server memory usage
- **CPU/GPU**: Monitor utilization

## Expected Behavior

1. **First Putt**: 
   - Green path (#22c55e)
   - Completes when ball stops for 0.5 seconds
   - Shows in results panel

2. **Second Putt**:
   - Blue path (#3b82f6)
   - Same completion logic

3. **Third Putt**:
   - Orange path (#f59e0b)
   - Session completes after this

4. **Completion**:
   - Status shows "All 3 putts completed!"
   - All paths remain visible
   - Summary shows all 3 putts

## Debug Mode

Enable debug logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)  # Change from INFO to DEBUG
```

This will show detailed information about:
- Frame processing
- Detection results
- Trajectory updates
- Putt completion logic

