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

### Usage example

```python
from model import UnifiedResNetModel

# Create OCR model
model = UnifiedResNetModel(head_type='ocr', pretrained=True)

# Forward pass
outputs = model(images)  # outputs is a list with predictions for each character
