# Pharmacy_Object_Detection_Using_YOLO
Sure! Here's a more detailed and elaborated version of the code explanation that you can use for a GitHub README file:

---

# YOLOv5 Custom Object Detection Pipeline

This repository demonstrates a custom object detection workflow using the YOLOv5 model for training, inference, and post-processing. The code involves loading a pre-trained YOLOv5 model, training it with custom data, and using the trained model to make predictions and visualize the results.

## Setup

### Step 1: Install YOLOv5 and Required Libraries

Ensure that the `ultralytics` package is installed to use the YOLO model for training and inference. Run the following command to install the package:

```bash
!pip install ultralytics
```

### Step 2: Load the YOLOv5 Model and Configuration

In this example, we use the YOLOv5 model. The YAML configuration file contains model hyperparameters and configurations.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml").load("yolo11n.pt")
model
```

### Step 3: Mount Google Drive (Optional)

If you're using Google Colab, you can mount your Google Drive to access datasets stored there.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Train the YOLOv5 Model

We train the model on a custom dataset using the following parameters:
- **Data:** Path to the YAML file with dataset details.
- **Epochs:** Number of training iterations.
- **Batch:** Number of images per batch.
- **Image Size (imgsz):** Image input size.
- **Learning Rate (lr0):** Initial learning rate.
- **Optimizer:** Optimizer to be used for training.
- **Momentum:** Momentum for gradient descent optimization.

```python
model.train(
    data = r"/content/drive/MyDrive/DS Project/22. Par Inventory Train/datasets/data.yaml", 
    epochs = 10,
    batch = 2, 
    imgsz = 800, 
    lr0 = 0.001, 
    optimizer = "adam", 
    momentum = 0.9
)
```

## Inference

### Step 5: Load the Trained Model

Load the trained YOLOv5 model for inference from the saved model weights.

```python
model = YOLO(r"/content/runs/detect/train3/weights/last.pt")
```

### Step 6: Load and Display an Image

Using OpenCV, we read and display an image from the test dataset:

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread(r"/content/drive/MyDrive/DS Project/22. Par Inventory Train/datasets/test/images/frame_0001_jpg.rf.94df2a1c04de8cc4c8aff4803c0a2e01.jpg")
cv2_imshow(img)  # Displays the image in Colab
```

### Step 7: Make Predictions with the YOLOv5 Model

We can make predictions on the image using the loaded YOLOv5 model. Here, we set confidence (`conf`) and IoU thresholds for Non-Max Suppression (NMS):

```python
results = model.predict(source=img, conf=0, iou=1)
```

### Step 8: Access and Process Predictions

Extract predicted bounding boxes, class labels, and confidence scores from the YOLO model results:

```python
result = results[0]
class_dict = result.names
boxes = result.boxes.xyxy
classes = result.boxes.cls
confidences = result.boxes.conf
```

### Step 9: Visualize Predictions on the Image

Draw bounding boxes and labels on the original image using OpenCV, based on the predictions:

```python
img_raw = img.copy()
for each_box, each_class, each_conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = each_box.cpu().data.numpy().astype("int")
    class_idx = int(each_class.item())
    class_name = class_dict[class_idx]
    cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img_raw, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2_imshow(img_raw)
```

## Post-Processing: Confidence Filtering and NMS (Non-Maximum Suppression)

### Step 10: Filter Predictions Based on Confidence

Set a confidence threshold to filter out lower-confidence predictions, and visualize the filtered boxes:

```python
conf_threshold = 0.4
final_boxes, final_classes, final_confidences = [], [], []
img_filtered = img.copy()

for each_box, each_class, each_conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = each_box.cpu().data.numpy().astype("int")
    class_idx = int(each_class.item())
    each_conf = each_conf.item()
    if each_conf > conf_threshold:
        final_boxes.append([x1, y1, x2, y2])
        final_classes.append(class_idx)
        final_confidences.append(each_conf)
        cv2.rectangle(img_filtered, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_filtered, class_dict[class_idx], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2_imshow(img_filtered)
```

### Step 11: Apply Class-Wise NMS

The Non-Maximum Suppression (NMS) function reduces redundant bounding boxes that highly overlap, leaving only the best ones based on confidence:

```python
def compute_iou(current_box, boxes):
    intersect_x1 = np.maximum(current_box[0], boxes[:, 0])
    intersect_y1 = np.maximum(current_box[1], boxes[:, 1])
    intersect_x2 = np.minimum(current_box[2], boxes[:, 2])
    intersect_y2 = np.minimum(current_box[3], boxes[:, 3])
    int_width = intersect_x2 - intersect_x1
    int_height = intersect_y2 - intersect_y1
    intersect_area = (int_width * int_height)

    current_box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
    boxes_area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    union_area = current_box_area + boxes_area - intersect_area
    return intersect_area / union_area

def class_wise_nms(boxes, classes, confidences, iou_threshold=0.6):
    unique_classes = np.unique(classes)
    boxes = np.array(boxes)
    classes = np.array(classes)
    confidences = np.array(confidences)

    nms_boxes, nms_classes, nms_confidences = [], [], []

    for each_class in unique_classes:
        class_boxes = boxes[classes == each_class]
        class_conf = confidences[classes == each_class]

        while len(class_boxes) > 0:
            max_index = np.argmax(class_conf)
            nms_boxes.append(class_boxes[max_index])
            nms_classes.append(each_class)
            nms_confidences.append(class_conf[max_index])

            current_box = class_boxes[max_index]
            iou = compute_iou(current_box, class_boxes)
            overlapping_idx = np.where(iou > iou_threshold)[0]

            class_boxes = np.delete(class_boxes, overlapping_idx, axis=0)
            class_conf = np.delete(class_conf, overlapping_idx)

    return nms_boxes, nms_classes, nms_confidences

nms_boxes, nms_classes, nms_confidences = class_wise_nms(final_boxes, final_classes, final_confidences)
```

### Step 12: Visualize NMS Results

The final visualization after applying NMS is done using OpenCV:

```python
img_final = img.copy()
for each_box, each_class, each_conf in zip(nms_boxes, nms_classes, nms_confidences):
    x1, y1, x2, y2 = each_box
    class_name = class_dict[each_class]
    cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img_final, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2_imshow(img_final)
```

---

### Conclusion

This repository provides a step-by-step guide to training a YOLOv5 model on a custom dataset and making predictions with confidence filtering and Non-Maximum Suppression. The post-processing ensures accurate and non-overlapping object detections, making the workflow suitable for real-time object detection tasks.

---

You can customize this README further with specific details about the project, such as the dataset used, performance metrics, or links to references for better understanding.
