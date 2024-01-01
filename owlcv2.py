import requests
from PIL import Image, ImageDraw
import torch
import numpy as np
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


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])

image_array = np.array(image)   
print(image_array.shape)
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(box)
    pt1 = (int(box[0]),int(box[1]))
    pt2 = (int(box[2]),int(box[3]))
    cv2.rectangle(image_array,pt1,pt2,(0,255,0),3)
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
cv2.imshow("image",image_array)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()