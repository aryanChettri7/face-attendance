import cv2
from ultralytics import YOLO

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO("yolov8n.pt")  # YOLOv8 nano model

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        results = self.model(frame)
        detections = results[0].boxes.data

        count = 0
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # 0 is 'person' in COCO
                count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(frame, f"People Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
