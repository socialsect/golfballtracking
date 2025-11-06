from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, List
import asyncio
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import json
import hashlib
import os
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25

# Session management
sessions: Dict[str, 'GolfBallSession'] = {}

@dataclass
class BallPoint:
    """Point in golf ball trajectory"""
    x: float
    y: float
    timestamp: float
    confidence: float

@dataclass
class Putt:
    """Completed putt with trajectory"""
    putt_number: int
    points: List[BallPoint]
    start_time: float
    end_time: float
    duration: float

class SessionState(Enum):
    WAITING = "waiting"
    DETECTING = "detecting"
    COMPLETED = "completed"

class GolfBallSession:
    """Session tracking 3 putts"""
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.state = SessionState.WAITING
        self.putts: List[Putt] = []
        self.current_trajectory: List[BallPoint] = []
        self.last_detection_time = None
        self.frame_count = 0
        self.stillness_frames = 0
        self.last_ball_position = None
        
        # Parameters for detecting putt completion
        self.STILLNESS_THRESHOLD = 0.5  # seconds of stillness to end putt
        self.MIN_PUTT_POINTS = 5  # minimum points for valid putt
        self.MAX_PUTT_DURATION = 10.0  # max seconds per putt
        
    def add_detection(self, x: float, y: float, confidence: float, timestamp: float):
        """Add a new ball detection point"""
        self.frame_count += 1
        
        # Check if ball moved significantly
        if self.last_ball_position:
            dx = x - self.last_ball_position[0]
            dy = y - self.last_ball_position[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # If ball moved, reset stillness counter
            if distance > 10:  # 10 pixels threshold
                self.stillness_frames = 0
                point = BallPoint(x=x, y=y, timestamp=timestamp, confidence=confidence)
                self.current_trajectory.append(point)
                self.last_ball_position = (x, y)
                self.last_detection_time = timestamp
            else:
                # Ball hasn't moved much
                self.stillness_frames += 1
        else:
            # First detection
            point = BallPoint(x=x, y=y, timestamp=timestamp, confidence=confidence)
            self.current_trajectory.append(point)
            self.last_ball_position = (x, y)
            self.last_detection_time = timestamp
            self.stillness_frames = 0
            
        # Check if putt should be completed
        if self.state == SessionState.DETECTING:
            self._check_putt_completion(timestamp)
    
    def _check_putt_completion(self, current_time: float):
        """Check if current putt should be finalized"""
        if not self.current_trajectory:
            return
            
        # Check stillness duration
        if self.last_detection_time:
            stillness_duration = current_time - self.last_detection_time
            if stillness_duration >= self.STILLNESS_THRESHOLD and len(self.current_trajectory) >= self.MIN_PUTT_POINTS:
                self._finalize_putt()
                return
        
        # Check max duration
        if self.current_trajectory:
            putt_duration = current_time - self.current_trajectory[0].timestamp
            if putt_duration >= self.MAX_PUTT_DURATION:
                self._finalize_putt()
                return
    
    def _finalize_putt(self):
        """Finalize current putt and start next one"""
        if len(self.current_trajectory) >= self.MIN_PUTT_POINTS:
            putt = Putt(
                putt_number=len(self.putts) + 1,
                points=self.current_trajectory.copy(),
                start_time=self.current_trajectory[0].timestamp,
                end_time=self.current_trajectory[-1].timestamp,
                duration=self.current_trajectory[-1].timestamp - self.current_trajectory[0].timestamp
            )
            self.putts.append(putt)
            logger.info(f"Putt {len(self.putts)} completed with {len(putt.points)} points")
        
        # Reset for next putt
        self.current_trajectory = []
        self.last_ball_position = None
        self.stillness_frames = 0
        self.last_detection_time = None
        
        # Check if all 3 putts are done
        if len(self.putts) >= 3:
            self.state = SessionState.COMPLETED
        else:
            self.state = SessionState.DETECTING
    
    def start_detection(self):
        """Start detecting putts"""
        self.state = SessionState.DETECTING
        self.current_trajectory = []
        self.last_ball_position = None
    
    def get_summary(self):
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "putts_completed": len(self.putts),
            "total_putts": 3
        }

def load_model():
    """Load YOLO model"""
    global model
    try:
        # Try multiple possible locations for the model
        model_paths = ['best.pt', 'weights/best.pt', '../best.pt']
        model = None
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = YOLO(path, task='detect')
                    logger.info(f"Model loaded from {path} on {device}")
                    break
            except Exception:
                continue
        
        if model is None:
            # Try default location
            model = YOLO('best.pt', task='detect')
            logger.info(f"Model loaded from default location on {device}")
        
        model.to(device)
        
        # Warmup
        dummy_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        _ = model(dummy_img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        logger.info("Model warmed up")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

@app.on_event("startup")
def startup():
    load_model()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image from bytes"""
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")
    
    # Resize to target size
    if frame.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
        frame = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    
    return frame

def detect_golf_ball(frame: np.ndarray) -> Optional[tuple]:
    """Detect golf ball in frame, returns (center_x, center_y, confidence) or None"""
    if model is None:
        return None
    
    try:
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=TARGET_SIZE, max_det=1)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Get first (best) detection
                box = boxes.xyxy[0].cpu().numpy()
                conf = float(boxes.conf[0].cpu().numpy())
                
                # Calculate center
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                return (center_x, center_y, conf)
    except Exception as e:
        logger.error(f"Detection error: {e}")
    
    return None

async def process_frame(session_id: str, image_bytes: bytes) -> dict:
    """Process a frame and return detection results"""
    session = sessions.get(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}
    
    try:
        # Preprocess
        frame = preprocess_image(image_bytes)
        timestamp = time.time()
        
        # Detect golf ball
        detection = detect_golf_ball(frame)
        
        response = {
            "success": True,
            "session_id": session_id,
            "state": session.state.value,
            "putts_completed": len(session.putts),
            "detected": False,
            "path_points": []
        }
        
        # If ball detected, add to trajectory
        if detection:
            x, y, conf = detection
            session.add_detection(x, y, conf, timestamp)
            response["detected"] = True
            response["ball_position"] = {"x": float(x), "y": float(y), "confidence": float(conf)}
        
        # Get path points for current trajectory
        if session.current_trajectory:
            response["path_points"] = [
                {"x": float(p.x), "y": float(p.y), "timestamp": p.timestamp}
                for p in session.current_trajectory
            ]
        
        # Get all completed putts
        response["putts"] = []
        for putt in session.putts:
            response["putts"].append({
                "putt_number": putt.putt_number,
                "points": [{"x": float(p.x), "y": float(p.y)} for p in putt.points],
                "duration": putt.duration,
                "point_count": len(putt.points)
            })
        
        # Check if completed
        if session.state == SessionState.COMPLETED:
            response["completed"] = True
            response["summary"] = session.get_summary()
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"success": False, "error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id: Optional[str] = None
    
    try:
        # Initial handshake
        init_msg = await websocket.receive_text()
        try:
            init_data = json.loads(init_msg)
        except:
            await websocket.send_text(json.dumps({"success": False, "error": "invalid_init"}))
            await websocket.close()
            return
        
        # Create or resume session
        if init_data.get("type") == "resume" and init_data.get("session_id"):
            sid = init_data["session_id"]
            if sid in sessions:
                session_id = sid
            else:
                # Create new session
                session = GolfBallSession(
                    session_id=f"session_{int(time.time() * 1000)}",
                    user_id=init_data.get("user_id", "anonymous")
                )
                sessions[session.session_id] = session
                session_id = session.session_id
        else:
            # Create new session
            session = GolfBallSession(
                session_id=f"session_{int(time.time() * 1000)}",
                user_id=init_data.get("user_id", "anonymous")
            )
            sessions[session.session_id] = session
            session_id = session.session_id
        
        # Send session started
        await websocket.send_text(json.dumps({
            "type": "started",
            "session_id": session_id,
            "state": sessions[session_id].state.value
        }))
        
        # Main loop
        while True:
            msg = await websocket.receive()
            
            if msg.get("type") == "websocket.receive":
                if "bytes" in msg and msg["bytes"]:
                    # Process frame
                    image_bytes = msg["bytes"]
                    response = await process_frame(session_id, image_bytes)
                    await websocket.send_text(json.dumps(response))
                    
                elif "text" in msg and msg["text"]:
                    # Control messages
                    try:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "start":
                            sessions[session_id].start_detection()
                            await websocket.send_text(json.dumps({
                                "type": "started",
                                "session_id": session_id,
                                "state": sessions[session_id].state.value
                            }))
                        elif ctrl.get("type") == "stop":
                            await websocket.send_text(json.dumps({
                                "type": "stopped",
                                "session_id": session_id
                            }))
                            break
                    except:
                        pass
            elif msg.get("type") == "websocket.disconnect":
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"success": False, "error": str(e)}))
        except:
            pass
    finally:
        # Cleanup (optional: keep session for a while)
        pass

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "active_sessions": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

