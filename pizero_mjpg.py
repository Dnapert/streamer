from picamera2 import Picamera2
from flask import Flask, Response
import cv2

app = Flask(__name__)

cam = Picamera2()
def generate_frames():
    cam = cam.start()
    while True:
        image = cam.capture_array()
        if not image:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate video frames

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)