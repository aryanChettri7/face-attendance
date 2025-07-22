import cv2
from ultralytics import YOLO
import numpy as np

class VideoCamera:
    def __init__(self):
        # Try to use camera first, then fallback to dummy frames
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Fallback to a dummy frame if camera is not available
            self.cap = None
        self.model = YOLO("yolov8n.pt")  # YOLOv8 nano model

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        # If camera is not available, return a dummy frame
        if self.cap is None:
            return self.create_dummy_frame()
        
        ret, frame = self.cap.read()
        if not ret:
            return self.create_dummy_frame()

        try:
            results = self.model(frame)
            
            # Check if results contain any detections
            if len(results) > 0 and results[0].boxes is not None:
                detections = results[0].boxes.data
                
                count = 0
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 0:  # 0 is 'person' in COCO
                        count += 1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                count = 0

            cv2.putText(frame, f"People Count: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                return jpeg.tobytes()
            else:
                return self.create_dummy_frame()
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return self.create_dummy_frame()
    
    def create_dummy_frame(self):
        # Create a dummy frame when camera is not available
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Camera not available", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(dummy_frame, "Running in demo mode", (170, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', dummy_frame)
        return jpeg.tobytes() if ret else b''
