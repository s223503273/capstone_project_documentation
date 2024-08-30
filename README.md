

# YOLOv10 Face Mask Detection

This repository contains the code and instructions for training a YOLOv10 model to detect whether people are wearing face masks correctly, incorrectly, or not at all. The project uses the "Face Mask Detection" dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Inference](#inference)
6. [Results](#results)


## Introduction
This project aims to detect three classes in images:
- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

The model is based on YOLOv10, a state-of-the-art object detection algorithm.

## Environment Setup
To replicate this project, you'll need to install the required packages and set up the environment.

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/yolov10-face-mask-detection.git
    cd yolov10-face-mask-detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download YOLOv10 weights:
    ```bash
    mkdir -p weights
    weights=('yolov10n.pt' 'yolov10s.pt' 'yolov10m.pt' 'yolov10b.pt' 'yolov10x.pt')
    base_url='https://github.com/THU-MIG/yolov10/releases/download/v1.1/'
    for weight in "${weights[@]}"; do
        wget -P weights -q "${base_url}${weight}"
    done
    ```

## Dataset Preparation
The dataset needs to be organized into training, validation, and test sets. Here's how to prepare the dataset:

1. **Create the necessary folder structure**:
    ```python
    def create_folder_structure(base_path):
        folders = [
            'data/train/images', 'data/train/labels',
            'data/val/images', 'data/val/labels',
            'data/test/images', 'data/test/labels'
        ]
        for folder in folders:
            os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    create_folder_structure('path_to_your_project')
    ```

2. **Convert XML annotations to YOLO format**:
    ```python
    def convert_annotation(xml_path, output_path, classes):
        # Your implementation here
    ```

3. **Process and split the dataset**:
    ```python
    process_dataset('path_to_your_dataset', 'path_to_output')
    ```

## Training the Model
Train the YOLOv10 model with the prepared dataset.

```bash
yolo task=detect mode=train epochs=50 batch=32 plots=True \
model=weights/yolov10n.pt \
data=data/data.yaml
```

Training logs and metrics will be saved in the `runs/detect/train/` directory.

## Inference
To perform inference on a new image using the trained model:

```python
from ultralytics import YOLOv10
import supervision as sv

model = YOLOv10('runs/detect/train/weights/best.pt')

# Load a random image from the test set
# Perform inference and annotate the image
```

## Results
The model's performance metrics such as Precision, Recall, and mAP50 are logged during training. Visual results of the predictions are saved in the `runs/detect/train/` directory.

)
