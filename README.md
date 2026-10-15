# APAN 5560 - Assignment 2: CNN

## Overview

This project extends Assignment 1 by adding a CNN image classifier
to the existing FastAPI application.

The CNN is built using PyTorch and trained on the CIFAR-10 dataset.

The application has two main endpoints:
- `/embedding`: Generate word embeddings using spaCy.
- `/classify`: Classify uploaded images using the CNN.

## CNN Model

The CNN follows the architecture given in Assignment 2.

- Input: 64 x 64 RGB image
- Conv2d: 16 filters, kernel size 3, stride 1, padding 1
- ReLU + MaxPooling (2 x 2)
- Conv2d: 32 filters, kernel size 3, stride 1, padding 1
- ReLU + MaxPooling (2 x 2)
- Flatten
- Fully connected layer: 8192 to 100
- ReLU
- Output layer: 100 to 10

The model predicts one of the 10 CIFAR-10 classes.

## Training Results

Dataset: CIFAR-10

Training settings:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 5
- Loss function: CrossEntropyLoss

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 1.4081 | 49.71% |
| 2 | 1.1070 | 60.55% |
| 3 | 0.9571 | 66.27% |
| 4 | 0.8287 | 70.80% |
| 5 | 0.7104 | 74.78% |

Final Test Loss: **1.0260**

Final Test Accuracy: **64.97%**

The trained model is saved in `models/cnn_model.pth`.

## How to Run

### 1. Build the Docker Image

```bash
docker build -t apan5560-assignment2 .
```

### 2. Run the Docker Container

```bash
docker run --rm -p 8000:80 apan5560-assignment2
```

### 3. Open FastAPI

Go to:

http://127.0.0.1:8000/docs

## API Testing

### POST /embedding

Enter a word to generate its embedding vector.

Example:

```json
{
  "word": "hello"
}
```

### POST /classify

1. Select `/classify` in FastAPI.
2. Click "Try it out".
3. Upload a JPG or PNG image.
4. Click "Execute".

The API returns the predicted class and confidence score.

Uploaded images are resized to 64 x 64 before classification.

## Model Training

To retrain the CNN:

```bash
uv sync
uv run python train.py
```