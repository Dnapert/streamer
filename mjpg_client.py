import cv2
import urllib.request as urllib 
import numpy as np

stream = cv2.VideoCapture('http://192.168.0.183:5000/video')
while True:
    ret,frame = stream.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Client',frame)
    if cv2.waitKey(1) == 27:
        break