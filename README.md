# Traffic Sign Detection using Faster R-CNN

Computer vision application for traffic sign detection using Faster R-CNN, PyTorch and custom annotated datasets.
The application uses a Faster R-CNN model built with PyTorch to detect and classify traffic signs from real-world street images using object detection and bounding box annotations.

This project was developed as part of the *Machine Learning & Deep Learning* course at Teknikhögskolan Göteborg.
This repository was originally created on a school-linked GitHub account and has been reuploaded to preserve the project in an active portfolio.

⚠️ This is an older project and does not fully reflect my current coding standards or development workflow, but it represents one of my earlier deep learning and computer vision projects.

---

## Tech Stack

- Python
- PyTorch
- Faster R-CNN
- Computer Vision
- Object Detection
- Deep Learning

---

## Features

- Traffic sign detection using Faster R-CNN
- Object localization with bounding boxes
- Multi-class traffic sign classification
- Custom annotated image datasets
- Deep learning inference pipeline
- Real-world street image predictions

---

## Dataset & Annotation

The project was trained on traffic sign images using manually annotated bounding boxes for object detection tasks.

The annotation process focused on:
- Traffic sign localization
- Multi-class detection
- Real-world street environments
- Detection accuracy across varying distances and lighting conditions

---

## Model Architecture

The project uses a Faster R-CNN architecture implemented in PyTorch for object detection and classification.

The model was trained to:
- Detect traffic signs in street images
- Generate bounding boxes around detected objects
- Predict traffic sign classes with confidence scores

---

## Example Prediction

Example of object detection and classification inference:

![Traffic Sign Detection](application/src/presentation/img/prediction_example.png)

---

## Challenges & Learnings

During development we worked with several important deep learning and computer vision concepts, including:

- Custom dataset annotation workflows
- Object detection pipelines
- Bounding box localization
- Multi-class classification
- Model generalization
- Deep learning inference
- Training stability and performance tuning

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
python app.py
```

---

## Contributors

This project was developed as a group project together with:

- Abshir
- Andreas
- Fredrik
- Kajsa

---

## Course Information

Machine Learning & Deep Learning  
Teknikhögskolan Göteborg  
2022
