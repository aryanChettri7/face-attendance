
import cv2
from ultralytics import YOLO
import numpy as np
import os
import json
from datetime import datetime
import uuid

class VideoCamera:
    def __init__(self):
        # Initialize YOLO model first
        self.model = YOLO("yolov8n.pt")  # YOLOv8 nano model
        
        # Attendance tracking variables
        self.attendance_log = []
        self.active_persons = {}  # Track persons currently in frame
        self.person_tracker = {}  # Simple tracking based on position
        self.next_person_id = 1
        self.tracking_threshold = 100  # Distance threshold for tracking
        self.attendance_file = "attendance_log.json"
        
        # Load existing attendance log
        self.load_attendance_log()
        
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

    def load_attendance_log(self):
        """Load existing attendance log from file"""
        try:
            if os.path.exists(self.attendance_file):
                with open(self.attendance_file, 'r') as f:
                    self.attendance_log = json.load(f)
        except Exception as e:
            print(f"Error loading attendance log: {e}")
            self.attendance_log = []

    def save_attendance_log(self):
        """Save attendance log to file"""
        try:
            with open(self.attendance_file, 'w') as f:
                json.dump(self.attendance_log, f, indent=2)
        except Exception as e:
            print(f"Error saving attendance log: {e}")

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def track_persons(self, detections):
        """Simple person tracking based on position"""
        current_persons = {}
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > 0.5:  # Person with good confidence
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Try to match with existing tracked persons
                matched = False
                for person_id, person_data in self.person_tracker.items():
                    if person_id not in current_persons:
                        distance = self.calculate_distance(
                            (center_x, center_y), 
                            person_data['last_position']
                        )
                        if distance < self.tracking_threshold:
                            current_persons[person_id] = {
                                'last_position': (center_x, center_y),
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(conf),
                                'last_seen': datetime.now().isoformat()
                            }
                            matched = True
                            break
                
                # If no match found, create new person
                if not matched:
                    person_id = f"Person_{self.next_person_id}"
                    self.next_person_id += 1
                    current_persons[person_id] = {
                        'last_position': (center_x, center_y),
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'first_seen': datetime.now().isoformat(),
                        'last_seen': datetime.now().isoformat()
                    }
                    
                    # Log new person entry
                    self.log_attendance(person_id, "ENTRY")
        
        # Check for persons who left the frame
        for person_id in self.person_tracker:
            if person_id not in current_persons:
                self.log_attendance(person_id, "EXIT")
        
        # Update tracker
        self.person_tracker = current_persons
        return current_persons

    def log_attendance(self, person_id, event_type):
        """Log attendance event"""
        event = {
            'person_id': person_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S')
        }
        self.attendance_log.append(event)
        self.save_attendance_log()
        print(f"Attendance logged: {person_id} - {event_type}")

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
                
                # Track persons and get current count
                current_persons = self.track_persons(detections)
                count = len(current_persons)
                
                # Draw bounding boxes and person IDs
                for person_id, person_data in current_persons.items():
                    x1, y1, x2, y2 = person_data['bbox']
                    conf = person_data['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw person ID and confidence
                    label = f"{person_id}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw center point
                    center_x, center_y = person_data['last_position']
                    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
            else:
                count = 0

            # Display attendance information
            cv2.putText(frame, f"Live Count: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            today_entries = len([log for log in self.attendance_log 
                               if log['date'] == datetime.now().strftime('%Y-%m-%d') 
                               and log['event_type'] == 'ENTRY'])
            
            cv2.putText(frame, f"Today's Entries: {today_entries}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

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
        
        cv2.putText(dummy_frame, "ATTENDANCE SYSTEM - DEMO MODE", (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(dummy_frame, "Running in Cloud Environment", (140, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dummy_frame, "Upload images or use local setup", (130, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(dummy_frame, "Live Count: 0 (Demo)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(dummy_frame, "Today's Entries: 0", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', dummy_frame)
        return jpeg.tobytes() if ret else b''
    
    def get_attendance_summary(self):
        """Get attendance summary for today"""
        today = datetime.now().strftime('%Y-%m-%d')
        today_logs = [log for log in self.attendance_log if log['date'] == today]
        
        entries = [log for log in today_logs if log['event_type'] == 'ENTRY']
        exits = [log for log in today_logs if log['event_type'] == 'EXIT']
        
        return {
            'total_entries': len(entries),
            'total_exits': len(exits),
            'current_occupancy': len(self.person_tracker),
            'recent_logs': today_logs[-10:] if today_logs else []
        }
    
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
