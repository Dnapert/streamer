import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

stream = cv2.VideoCapture('http://192.168.0.183:5000/video')
while True:
    try:
        ret,frame = stream.read()
    except:
        pass
    if ret:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(img_rgb,conf=.5,verbose=False,device=1,save=False,show=True)
        if cv2.waitKey(1) == 27:
            break
stream.release()