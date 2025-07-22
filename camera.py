import cv2
from ultralytics import YOLO
import numpy as np
import os

class VideoCamera:
    def __init__(self):
        # Initialize YOLO model first
        self.model = YOLO("yolov8n.pt")  # YOLOv8 nano model
        
        # Try multiple camera sources
        self.cap = None
        self.camera_available = False
        
        # Try different camera indices
        for i in range(4):  # Try camera indices 0, 1, 2, 3
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.camera_available = True
                    print(f"Camera found at index {i}")
                    break
                cap.release()
        
        # If no camera found, try different backends
        if not self.camera_available:
            backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG]
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(0, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.cap = cap
                            self.camera_available = True
                            print(f"Camera found with backend {backend}")
                            break
                    cap.release()
                except:
                    continue
        
        if not self.camera_available:
            print("No camera available - using demo mode")
            self.cap = None

    def __del__(self):
        if self.cap is not None:
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
        
        # Add background pattern for better visibility
        for i in range(0, 640, 40):
            cv2.line(dummy_frame, (i, 0), (i, 480), (30, 30, 30), 1)
        for i in range(0, 480, 40):
            cv2.line(dummy_frame, (0, i), (640, i), (30, 30, 30), 1)
        
        cv2.putText(dummy_frame, "WEBCAM NOT AVAILABLE", (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(dummy_frame, "Running in Cloud Environment", (140, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dummy_frame, "Upload images or use local setup", (130, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(dummy_frame, "People Count: 0 (Demo)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', dummy_frame)
        return jpeg.tobytes() if ret else b''
    
    def process_uploaded_image(self, image_path):
        """Process an uploaded image and save the result"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not read image: {image_path}")
                return
            
            results = self.model(frame)
            
            # Process detections
            count = 0
            if len(results) > 0 and results[0].boxes is not None:
                detections = results[0].boxes.data
                
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 0:  # 0 is 'person' in COCO
                        count += 1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {count}: {conf:.2f}", 
                                  (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(frame, f"People Count: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save processed image
            output_path = image_path.replace('.', '_processed.')
            cv2.imwrite(output_path, frame)
            print(f"Processed image saved to: {output_path}")
            print(f"Detected {count} people in the image")
            
        except Exception as e:
            print(f"Error processing uploaded image: {e}")
