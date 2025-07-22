from flask import Flask, render_template, Response, request, flash, redirect, url_for, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === CAMERA CLASS ===
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.attendance_log = []

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def process_uploaded_image(self, filepath):
        # Dummy processor - logs filename and timestamp
        self.attendance_log.append({
            'file': os.path.basename(filepath),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    def get_attendance_summary(self):
        return {
            'total_entries': len(self.attendance_log),
            'current_occupancy': len(self.attendance_log)
        }

    def save_attendance_log(self):
        with open("attendance_log.txt", "w") as f:
            for entry in self.attendance_log:
                f.write(f"{entry['file']} - {entry['timestamp']}\n")

camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance_dashboard():
    return render_template('attendance.html')

def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(f"[ERROR] Streaming frame failed: {e}")
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance_summary')
def attendance_summary():
    try:
        summary = camera.get_attendance_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance_logs')
def attendance_logs():
    try:
        return jsonify(camera.attendance_log[-50:])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        camera.process_uploaded_image(filepath)
        flash('Image uploaded and processed successfully!')
    else:
        flash('Please upload a valid image file (PNG, JPG, JPEG)')
    
    return redirect(url_for('index'))

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    try:
        camera.attendance_log = []
        camera.save_attendance_log()
        flash('Attendance logs cleared successfully!')
    except Exception as e:
        flash(f'Error clearing logs: {str(e)}')
    
    return redirect(url_for('attendance_dashboard'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
