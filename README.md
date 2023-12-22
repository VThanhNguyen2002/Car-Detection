# Introduction:
The "CarDetectionPython" project uses the Python programming language. The project employs the YOLOv5 model to classify cars in a labeled image dataset on the Roboflow platform. However, this project is still lacking a Region of Interest (ROI) method.
# Project Details
- Course: Deep Learning
- Language Used: Python

# YOLOv5 Car Detection Project Guide

## Step 1: Install YOLOv5 and Dependencies

```bash
pip install ultralytics
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5/
!pip install -r requirements.txt
!pip install roboflow
```

## Step 2: Fetch Data from Roboflow
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="mclLHXcZODPiI3gS6M3W")
project = rf.workspace("mit-ivmqi").project("car-detection-xpuvx")
dataset = project.version(3).download("yolov5")
```

## Step 3: Train the Model
```bash
!python /content/drive/MyDrive/yolov5/train.py --img 640 --batch 16 --epochs 50 --data /content/drive/MyDrive/yolov5/Car-detection-3/data.yaml --weights yolov5s.pt
```
Note on Parameters:
- img 640: Input image size is 640x640 pixels.
- batch 16: Batch size is 16.
-  epochs 50: Number of training epochs is 50.
-  data /content/drive/MyDrive/yolov5/Car-detection-3/data.yaml: Path to the data.yaml file containing dataset information.

## Step 4: Install Missing Requirements
```bash
pip install gitpython
```

If there are issues with requirements, run the following command:
```bash
pip install -r requirements.txt
```
If there is an error with gitpython, try:
```bash
pip install gitpython
```
Note: You may need to restart the runtime or rerun the command to update requirements.

## Step 5: View Results
After training, you will see the model results and statistics, along with instructions on how to use your trained model.

Note: Check the results and adjust parameters if needed. Also, ensure that you have installed the necessary libraries before running the commands.

## Step 6: Model Information

After training, you will see the model results and statistics, along with instructions on how to use your trained model.

**Note**: Check the results and adjust parameters if needed. Also, ensure that you have installed the necessary libraries before running the commands.

## Step 7: View Results and Evaluate the Model

```bash
# Results on the validation dataset
!python /content/drive/MyDrive/yolov5/val.py --weights /content/drive/MyDrive/yolov5/runs/train/exp4/weights/best.pt

# Evaluate the model
!python /content/drive/MyDrive/yolov5/test.py --weights /content/drive/MyDrive/yolov5/runs/train/exp4/weights/best.pt

```

## Step 8: Use the Trained Model
After training and evaluating the model, you can use your trained model to make predictions on new images or videos.
### Import YOLOv5 inference module
```bash
from pathlib import Path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync

# Load the trained model
weights = '/content/drive/MyDrive/yolov5/runs/train/exp4/weights/best.pt'
device = select_device('')
model = attempt_load(weights, map_location=device)

# Inference on a single image
img_path = '/path/to/your/image.jpg'
img = cv2.imread(img_path)[:, :, ::-1]
img = check_img_size(img, s=model.stride.max())
img = torch.from_numpy(img).to(device)
img = img.float() / 255.0
img = img.unsqueeze(0)

# Make predictions
pred = model(img)[0]

# Process predictions
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
pred = pred[0]

# Draw bounding boxes on the image
for det in pred:
    bbox = det[:4].int()
    cls_conf = det[4]
    cls_id = det[5]
    label = f'Class {int(cls_id)}: {cls_conf:.2f}'
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the result
plt.imshow(img.numpy())
plt.show()
```
Note: Adjust parameters such as conf_thres and iou_thres to ensure results fit your requirements.
## Step 9: Summary and Support
This is the final step to summarize your project, and you can add contact information, licenses, or more detailed instructions if needed. Additionally, you can include instructions for using ClearML or Comet for easy monitoring and visualization of your model results.

Best of luck with your YOLOv5 project!

## Step 10: Deploy the Model
If you want to deploy your YOLOv5 model for use in real applications, you may consider the following steps:
1. **Convert Model to ONNX Format (Optional):**
   ```bash
   # Convert model to ONNX format
   !python /content/drive/MyDrive/yolov5/export.py --img-size 640 --batch-size 1 --weights /content/drive/MyDrive/yolov5/runs/train/exp4/weights/best.pt - -include pb,torchscript,onnx

- weights yolov5s.pt: Path to the YOLOv5s model weights file.

Once you have an ONNX file, you can deploy the model in mobile apps or integrate into your services.

# Conclude
This is the ultimate guide to training and using the YOLOv5 model for your vehicle recognition project.
