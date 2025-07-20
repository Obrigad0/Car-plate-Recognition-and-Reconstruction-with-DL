# Car Plate Recognition and Reconstruction with Deep Learning

## Overview

This project tackles the problem of automatic vehicle license plate recognition using deep learning techniques. The system is designed to work on images captured in real-world scenarios, handling challenges such as varying lighting, occlusions, motion blur, and different plate formats.

The implemented architecture consists of two main stages:  
- **License plate detection:** using a YOLOv5 model for fast and accurate localization of the plate within the vehicle image.  
- **License plate recognition:** using a specialized PDLPR model to decode the sequence of alphanumeric characters on the plate.

Additionally, a **baseline model** based on a unified ResNet18 architecture is implemented, capable of performing both bounding box regression and multi-head OCR recognition.

## Dataset

- The [CCPD (Chinese City Parking Dataset)](https://github.com/detectRecog/CCPD) is used, containing vehicle images with diverse and challenging conditions.

## Baseline Architecture

The baseline is implemented with the `UnifiedResNetModel` PyTorch class:

- Backbone: ResNet18 (pretrained or trained from scratch)  
- Two possible head types:  
  - **bbox:** bounding box regression head predicting 4 coordinates  
  - **ocr:** multi-head OCR, with one classification head per character (7 characters per plate, 68 possible classes per character)

The model supports saving and loading backbone and head weights separately for modularity.

## YOLOv5 Installation

To use the license plate detection part of this project, please follow the official YOLOv5 repository guidelines for installation and setup. This ensures you have the necessary dependencies and environment configured correctly.

Typical steps include:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```
Make sure you have a compatible PyTorch version installed (YOLOv5 recommends PyTorch 1.7+).

You can then integrate the YOLOv5 detection model with the rest of this project for plate localization.

## PDLPR Model

The core recognition model in this project is the **PDLPR (Positional Deep License Plate Recognition)** network, which combines convolutional feature extraction with transformer-based encoder-decoder layers to accurately decode the alphanumeric characters on the detected license plates.

### Architecture Overview

- **IGFE (Improved General Feature Extractor):**  
  A custom convolutional backbone that includes a FocusStructure layer for efficient spatial downsampling, residual blocks, and downsampling layers to extract robust features from the input plate images.

- **Positional Encoding 2D:**  
  Injects spatial positional information into the feature maps, crucial for transformer attention mechanisms.

- **Encoder:**  
  Stacks of modules combining convolutional layers and multi-head self-attention to encode spatial features globally.

- **Decoder:**  
  Transformer-style decoding modules with self-attention, cross-attention over encoder outputs, and feed-forward layers to predict sequences of characters.

- **Classifier:**  
  Predicts character probabilities for each position in the plate sequence, handling 68 classes (letters, digits, and special characters).

### Model Usage

The PDLPR model is implemented as a PyTorch `nn.Module` named `PDLPR`. It expects input images with 3 channels (RGB) and outputs the predicted logits for the license plate characters.

Example instantiation:

```python
model = PDLPR(in_channels=3, base_channels=512, num_classes=68, seq_len=8)

# Forward pass
outputs = model(images)  # outputs contains predictions for each character
