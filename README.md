I'll use the code provided in your PDF to craft the GitHub documentation. Here's an updated README that incorporates the code directly from your document:

---

# YOLOv10 Face Mask Detection

This repository contains the code and instructions for training a YOLOv10 model to detect whether people are wearing face masks correctly, incorrectly, or not at all. The project uses the "Face Mask Detection" dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Inference](#inference)
6. [Scripts](#scripts)
7. [Results](#results)
8. [Conclusion](#conclusion)


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
    pip install -q git+https://github.com/THU-MIG/yolov10.git
    pip install -q supervision 
    pip uninstall -y wandb
    pip install -q --upgrade huggingface_hub
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
    import os
    
    def create_folder_structure(base_path):
        folders = [
            'data/train/images', 'data/train/labels',
            'data/val/images', 'data/val/labels',
            'data/test/images', 'data/test/labels'
        ]
        for folder in folders:
            os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    create_folder_structure(os.getcwd())
    ```

2. **Convert XML annotations to YOLO format**:
    ```python
    import xml.etree.ElementTree as ET
    
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        return (x * dw, y * dh, w * dw, h * dh)

    def convert_annotation(xml_path, output_path, classes):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        with open(output_path, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    ```

3. **Process and split the dataset**:
    ```python
    from sklearn.model_selection import train_test_split
    import shutil
    
    def process_dataset(dataset_path, output_path):
        classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        image_folder = os.path.join(dataset_path, 'images')
        annotation_folder = os.path.join(dataset_path, 'annotations')
        
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        
        # Split the dataset
        train_val, test = train_test_split(image_files, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.2, random_state=42)
        
        # Process and move files
        for split, files in [('train', train), ('val', val), ('test', test)]:
            for file in files:
                # Image
                src_img = os.path.join(image_folder, file)
                dst_img = os.path.join(output_path, f'data/{split}/images', file)
                shutil.copy(src_img, dst_img)
                
                # Annotation
                xml_file = os.path.splitext(file)[0] + '.xml'
                src_xml = os.path.join(annotation_folder, xml_file)
                dst_txt = os.path.join(output_path, f'data/{split}/labels', os.path.splitext(file)[0] + '.txt')
                convert_annotation(src_xml, dst_txt, classes)
        
        # Create data.yaml
        yaml_content = {
            'train': f'{output_path}/data/train/images',
            'val': f'{output_path}/data/val/images',
            'test': f'{output_path}/data/test/images',
            'nc': len(classes),
            'names': classes
        }
        
        with open(os.path.join(output_path, 'data', 'data.yaml'), 'w') as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    process_dataset('/kaggle/input/face-mask-detection', os.getcwd())
    ```

## Training the Model
Train the YOLOv10 model with the prepared dataset.

```bash
yolo task=detect mode=train epochs=50 batch=32 plots=True \
model=$(pwd)/weights/yolov10n.pt \
data=$(pwd)/data/data.yaml
```

Training logs and metrics will be saved in the `runs/detect/train/` directory.

## Inference
To perform inference on a new image using the trained model:

1. **Load the trained model and perform prediction**:
    ```python
    from ultralytics import YOLOv10
    import supervision as sv
    import random
    
    # Load the trained model
    model = YOLOv10(f'{os.getcwd()}/runs/detect/train/weights/best.pt')

    # Load the dataset
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{os.getcwd()}/data/test/images",
        annotations_directory_path=f"{os.getcwd()}/data/test/labels",
        data_yaml_path=f"{os.getcwd()}/data/data.yaml"
    )

    # Randomly select an image from the test set
    random_item = random.choice(list(dataset))
    random_image_path, random_image, _ = random_item

    # Perform inference
    results = model(source=random_image, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the image
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = box_annotator.annotate(scene=random_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Display the result
    sv.plot_image(annotated_image)
    ```

## Scripts
This repository includes several scripts to help with dataset preparation, training, and inference:

1. **`prepare_dataset.py`**: Script for preparing and splitting the dataset.
    ```bash
    python prepare_dataset.py --dataset_path path_to_dataset --output_path path_to_output
    ```

2. **`train_model.py`**: Script to train the YOLOv10 model.
    ```bash
    python train_model.py --epochs 50 --batch_size 32 --weights weights/yolov10n.pt --data data/data.yaml
    ```

3. **`predict.py`**: Script to run inference on new images.
    ```bash
    python predict.py --model_path runs/detect/train/weights/best.pt --image_path path_to_image
    ```

### Example Command to Run a Script
```bash
python predict.py --model_path runs/detect/train/weights/best.pt --image_path data/test/images/example_image.png
```

## Results
The model's performance metrics such as Precision, Recall, and mAP50 are logged during training. Visual results of the predictions are saved in the `runs/detect/train/` directory.

## Conclusion
The trained YOLOv10 model successfully detects the presence and correctness of face masks. Further improvements could involve fine-tuning the model or increasing the dataset size.


