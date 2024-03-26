import cv2
import time
cap = cv2.VideoCapture(0)
w,h = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(w,h)
time.sleep(5)
cap.release()
