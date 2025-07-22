
from flask import Flask, render_template, Response, request, flash, redirect, url_for, jsonify
from camera import VideoCamera
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
            print(f"Error in video stream: {e}")
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance_summary')
def attendance_summary():
    """API endpoint to get attendance summary"""
    try:
        summary = camera.get_attendance_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance_logs')
def attendance_logs():
    """API endpoint to get all attendance logs"""
    try:
        return jsonify(camera.attendance_log[-50:])  # Return last 50 logs
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded image
        camera.process_uploaded_image(filepath)
        flash('Image uploaded and processed successfully!')
    else:
        flash('Please upload a valid image file (PNG, JPG, JPEG)')
    
    return redirect(url_for('index'))

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Clear all attendance logs"""
    try:
        camera.attendance_log = []
        camera.save_attendance_log()
        flash('Attendance logs cleared successfully!')
    except Exception as e:
        flash(f'Error clearing logs: {str(e)}')
    
    return redirect(url_for('attendance_dashboard'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
