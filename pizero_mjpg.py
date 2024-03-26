from picamera2 import Picamera2
from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames():
    cam = Picamera2()
    cam.start()
    while True:
        image = cam.capture_array()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', img_rgb)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate video frames

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)