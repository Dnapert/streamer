import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
class_names = model.names
stream = cv2.VideoCapture('http://192.168.0.183:5000/video')
while True:
    ret,frame = stream.read()
    if ret:
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(img_rgb, conf=.25, show=False, persist=True, save=False,verbose=False, device=0)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = box
                class_name = class_names[int(cls)]
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_rgb, f'{class_name} {track_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Stream', img_rgb)

        if cv2.waitKey(1) == 27:
            break