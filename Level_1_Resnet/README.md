# Custom ResNet on CIFAR-10

## Project Overview
This project implements a custom ResNet architecture from scratch using PyTorch for image classification on the CIFAR-10 dataset.

## Features
- Custom Residual Blocks
- Skip Connections
- Batch Normalization
- Global Average Pooling
- Model Training and Testing
- Confusion Matrix
- Classification Report

## Dataset
- CIFAR-10
- 50,000 training images
- 10,000 testing images
- 10 image classes

## Results
- Training Accuracy: 82.44%
- Test Accuracy: 79.54%

## Folder Structure
```
Level_1_Resnet/
├── code/
├── model_weights/
├── outputs/
└── README.md
```

## How to Run

```bash
python train.py
python test.py
```

## Framework
- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn