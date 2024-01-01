from PIL import Image
import torch
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
    
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

texts = [["a person", "a piano "]]

def get_cap():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    return cap

def draw_boxes(image_array,boxes,scores,labels):
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(box)
        pt1 = (int(box[0]),int(box[1]))
        pt2 = (int(box[2]),int(box[3]))
        cv2.rectangle(image_array,pt1,pt2,(0,255,0),3)
        cv2.putText(image_array,texts[0][label],pt1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        print(f"Detected {texts[0][label]} with confidence {round(score.item(), 3)} at location {box}")
    return image_array

def process_frames(cap:cv2.VideoCapture,model,texts):
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            t_image = Image.fromarray(image)
            inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.Tensor([t_image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
            scores = results[0]["scores"]
            boxes = results[0]["boxes"]
            labels = results[0]["labels"]
            image = draw_boxes(image,boxes,scores,labels)
        #press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow("image",image)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = get_cap()
    process_frames(cap,model,texts)
    cap.release()
    cv2.destroyAllWindows()

