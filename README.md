# TrackZone

## Introduction

This project introduces a robust human detection and tracking model designed to monitor individuals within a predefined zone in real time. The zone is manually created to specify the area of interest, and the model continuously tracks and counts the number of people present within this designated space.

Human detection is powered by advanced technologies, leveraging the **YOLOv8** framework for efficient object detection, **Roboflow** for dataset management and preprocessing, and the **Supervision** libraries for streamlined tracking and visualization. Together, these tools enable accurate and reliable live tracking, making the system highly effective for real-time surveillance and monitoring applications.

## Features

- Real-time human detection and tracking within a predefined zone.
- Customizable zone area for tracking based on user-defined coordinates.
- Uses YOLOv8 for efficient and accurate object detection.
- Easy integration with Roboflow for dataset management and preprocessing.
- Real-time tracking with the Supervision library for zone detection and visualization.
- Displays the total number of detections and zone-specific detections.

## Installation

-Install Python 3.8 or later.
-Install necessary libraries using pip:pip install ultralytics roboflow supervision opencv-python-headless matplotlib numpy
-Download the dataset via Roboflow and configure paths.
-Prepare the YOLOv8 model weights (e.g., yolov8n.pt) for training.

## Dataset
 Link for the dataset: [Link](https://universe.roboflow.com/dave-r7nyf/test_detection_person)
The dataset used in this project is managed and preprocessed through **Roboflow**. It contains images with annotations for human detection. The dataset can be downloaded from Roboflow, and the paths are configured within the project. The dataset includes the following:

- **Train Dataset:** Images for training the model.
- **Validation Dataset:** Images for validating the model performance.
- **Test Dataset:** Images used for testing and final evaluation.

## Model

The human detection and tracking system is built on **YOLOv8** (You Only Look Once version 8) for efficient and accurate object detection. The model is initialized with pre-trained weights (yolov8n.pt) and fine-tuned on a custom dataset. The training process uses the following parameters:
- 50 epochs
- Image size of 640x640

Once trained, the model is exported to the **ONNX** format for compatibility with different platforms.

## Training
- Training is conducted on the custom dataset provided by **Roboflow**.
- Includes hyperparameters like **epochs** and **image size** for optimization.
- The **best-performing weights** are saved and used for further evaluation.

## Evaluation
- Validation includes metrics such as **accuracy**, **precision**, and **recall**.
- Results are printed to assess model performance and make adjustments if necessary.

## Usage
- Use the trained **YOLOv8** model to analyze videos for object detection and tracking.
- Define specific **zones** in the video to focus on objects within those regions.
- Save the **annotated video outputs** with detection statistics.

## Results
- The model successfully detects objects and counts **zone-specific detections**.
- Annotated videos include **bounding boxes** and overlays for total and zone detections.
- Outputs are saved in the specified directory for further use.

## Challenges

### Finding the Best Dataset to Train the YOLO model for detections
One of the key challenges in building an effective human detection and tracking model is selecting the best dataset for training the YOLOv8 model. The quality of the dataset plays a critical role in the model's performance. Some of the difficulties involved include:

- **Data Relevance:** The dataset must contain diverse and relevant images that closely resemble the conditions under which the model will operate. This includes varying lighting conditions, different human poses, multiple people in the frame, and different background settings.
  
- **Data Annotation:** High-quality annotations are necessary for training the YOLO model. This means that the dataset should have accurate labels for each object (in this case, humans) with bounding boxes or segmentation masks. Poor or inconsistent annotations can lead to suboptimal performance.

- **Dataset Size:** A large and well-balanced dataset is often required to ensure the model generalizes well. Insufficient data or a dataset that is heavily imbalanced (e.g., more images of a particular class) can lead to overfitting, where the model performs well on the training data but poorly on unseen data.

- **Handling Edge Cases:** The dataset must also account for edge cases such as occlusions, people in unusual poses, or rare scenarios like partial detections. These edge cases are often difficult to capture, but they are crucial for creating a robust model.

- **Dataset Availability:** Finding a high-quality, publicly available dataset that fits the specific needs of the project can be challenging. While platforms like **Roboflow** provide access to curated datasets, there may still be cases where a custom dataset needs to be created, which involves considerable effort in data collection and annotation.

To overcome these challenges, a combination of publicly available datasets and custom data collection may be necessary to ensure the model is adequately trained and performs well in real-world scenarios.

## Future Scope of Improvement

### 1. Using a Better Dataset for Improved Model Training
One of the key areas for future improvement lies in the dataset used for training the YOLOv8 model. By utilizing a more diverse and high-quality dataset, the model's performance can be significantly enhanced. This can include datasets with better annotations, more varied environments, different human poses, and challenging scenarios such as occlusions and partial detections. A richer dataset would help the model generalize better, leading to higher accuracy and robustness in real-world applications.

### 2. Exploring YOLOv8 Variants (YOLOv8s, YOLOv8m, YOLOv8l)
Another potential improvement is experimenting with different versions of YOLOv8, such as **YOLOv8s**, **YOLOv8m**, and **YOLOv8l**. These models come in different sizes (small, medium, large) and offer trade-offs between speed and accuracy. By testing with different model versions, it is possible to find the most optimal configuration for the specific use case, balancing detection accuracy with processing time, especially in real-time tracking scenarios.

### 3. Testing Detected Objects with Detectron2
In the future, the model’s detection results can be further validated by testing them with **Detectron2**, a state-of-the-art object detection library developed by Facebook AI Research. Using Detectron2 for comparative evaluation will help assess how the YOLOv8 model performs relative to other advanced detection algorithms. This will allow for further fine-tuning of the model and enable the exploration of additional post-processing techniques for improving detection accuracy.

