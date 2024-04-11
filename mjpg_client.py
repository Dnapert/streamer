import cv2

stream = cv2.VideoCapture('http://192.168.0.183:5000/video')
while True:
    ret,frame = stream.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Client',img_rgb)
    if cv2.waitKey(1) == 27:
        break