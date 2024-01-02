from owl_clip.sort import *
from PIL import Image
import torch
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")
    
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

texts = [["a person", "a piano ","a cat","a television","a dog","computer","remote","keyboard","radio"]]

def get_cap():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    return cap

def draw_boxes(image_array,track_bboxes_ids):
    for boxes in track_bboxes_ids:
        box = boxes[:4]
        id = boxes[4]
        pt1 = (int(box[0]),int(box[1]))
        pt2 = (int(box[2]),int(box[3]))
        cv2.rectangle(image_array,pt1,pt2,(0,255,0),3)
        cv2.putText(image_array,str(id),pt1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    return image_array

def process_frames(cap:cv2.VideoCapture,model,texts):
    sort = Sort()
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            t_image = Image.fromarray(image)
            inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.Tensor([t_image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, threshold=0.4, target_sizes=target_sizes)
            scores = results[0]["scores"]
            boxes = results[0]["boxes"]
            result_list = []
            for box,score in zip(boxes,scores):
                box = [round(i, 2) for i in box.tolist()]
                box.append(round(score.item(), 3))
                result_list.append(box)
            #print(result_list)
            if len(result_list) > 0:
                track_bboxes_ids = sort.update(np.array(result_list))
                image = cv2.cvtColor(draw_boxes(image,track_bboxes_ids),cv2.COLOR_RGB2BGR)
            else:
                track_bboxes_ids = sort.update(np.empty((0,5)))      
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

