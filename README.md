# Cattle Detection using Detectron2

This repository contains code for detecting cattle in imported frames using the Detectron2 model. We utilized the Detectron2 framework, specifically the detectron2.ModelZoo's detectron2.model_zoo.get() function to fetch the pre-trained model, and we chose the detectron2.model_zoo.get("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") model for this purpose.

## Functions and Thresholding
We've designed two separate functions to accomplish the cattle detection task: one for obtaining masks and another for extracting bounding boxes. These functions help in capturing the distinct regions where cattle are detected in the frames.

Furthermore, we've set a specific threshold to fine-tune the accuracy of our detections. The threshold value was determined through experimentation and validation. This thresholding process aids in filtering out false positives and ensures that only confident predictions are retained.

## Class Filtering
Given the nature of our application, we focused on two specific classes: 'horse' and 'cattle'. The class numbers corresponding to these classes are 17 and 19, respectively. By concentrating on these classes, we aimed to improve the model's performance, particularly in scenarios where cattle might be misclassified as horses from certain angles, such as the caudal view.

## Confidence Filtering
To further enhance the accuracy of our results, we implemented a confidence threshold. Predictions with a confidence score below 95% were excluded from the final output. This ensures that only high-confidence detections are considered, reducing the chances of false positives and enhancing the overall quality of the results.

By combining these strategies – class filtering, confidence thresholding, and dedicated functions for masks and bounding boxes – we've developed a robust cattle detection system. This system serves as an efficient tool for identifying and extracting cattle regions from the input frames, contributing to a variety of applications such as livestock monitoring and research.

Feel free to explore the code and adapt it to your own projects. If you have any questions or suggestions, please don't hesitate to reach out.

# Transcript
```python
#installing the model
!python -m pip install pyyaml==5.1
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

```python
#Importing thetorch and detectron_2
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
```

```python
# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
from PIL import Image
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```
Function Purpose:
This function performs image segmentation, resulting in segmented images.

Note for Batch Processing:
If you intend to import multiple images, consider modifying the output naming in this section: {Save the ROI as a PNG file}.

```python

def image_segmentation(image):
  im = cv2.imread(image)
  cfg = get_cfg()

  # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

  # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)

  #Getting the predicted score, predicted mask, predicted class and predicted box
  pred_masks_scores = outputs["instances"].scores.cpu().numpy()
  pred_masks = outputs["instances"].pred_masks.detach().cpu().numpy()
  pred_class = outputs["instances"].pred_classes.detach().cpu().numpy()
  pred_box = outputs["instances"].pred_boxes.detach().cpu().numpy()
  valid_mask_indices = np.where((pred_masks_scores > 0.90) & (pred_class ==19))[0]

  #This code will detect several masks for each imputed image we get the valid masks that passed our threshold.
  for idx in valid_mask_indices:
        
        mask = pred_masks[idx]
        # Create a binary mask from the predicted mask
        binary_mask = (mask * 255).astype(np.uint8)

        # Resize the binary mask to match the image dimensions
        binary_mask_resized = cv2.resize(binary_mask, (im.shape[1], im.shape[0]))

        # Create a mask image that is True where the binary mask is active
        mask_bool = binary_mask_resized > 0

        # Extract the region of interest (ROI) from the original image based on the mask
        roi = im.copy()
        roi[~mask_bool] = 0  # Set non-masked regions to black

        # Save the ROI as a PNG file
        cv2.imwrite(f"roi_{idx}.png", roi)

  return pred_masks

```
Function for Bounding Boxes:
This function is designed to generate bounding boxes around detected objects.

Note about Multiple Masks:
Keep in mind that this function will produce multiple masks for each input image.

```python

def image_box(image, visualize=True):
    im = cv2.imread(image)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    pred_scores = outputs["instances"].scores.cpu().numpy()
    pred_boxes = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
    pred_class = outputs["instances"].pred_classes.detach().cpu().numpy()
    valid_box_indices = np.where((pred_scores > 0.95) & ((pred_class == 17) or (pred_class == 19)))[0]

    image_base_name = os.path.splitext(os.path.basename(image))[0]

    for idx in valid_box_indices:
        box = pred_boxes[idx]
        x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers

        # Extract the region of interest (ROI) from the original image based on the bounding box
        roi = im[y_min:y_max, x_min:x_max]

        # Specify the output directory and file name using the input image's name
        output_directory = '/content/171407'
        output_file_name = f"{image_base_name}_roi_{idx}.png"
        output_path = os.path.join(output_directory, output_file_name)
        cv2.imwrite(output_path, roi)

    return pred_scores

```
Here is the code snippet you can use to call these functions:

```python
# Directory path where your uploaded images are located
uploaded_image_directory = 'Your_directory_path'

image_files = os.listdir(uploaded_image_directory)
# List all files in the uploaded images directory

for image_file in image_files:
    image_path = os.path.join(uploaded_image_directory, image_file)

    # Open and process the image using Pillow (PIL)
    Call_your_fnction(image_path)
```



