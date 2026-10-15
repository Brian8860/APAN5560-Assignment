# APAN 5560 - Assignment 2: CNN

## Overview

This project extends Assignment 1 by adding a CNN image classifier
to the existing FastAPI application.

The CNN is built using PyTorch and trained on the CIFAR-10 dataset.

The application has two main endpoints:
- `/embedding`: Generate word embeddings using spaCy.
- `/classify`: Classify uploaded images using the CNN.

## CNN Model

The model uses:
- Two convolutional layers with ReLU and MaxPooling
- One fully connected hidden layer
- One output layer for 10 classes

The input images are 32 x 32 RGB images.

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
| 1 | 1.3645 | 50.90% |
| 2 | 0.9866 | 65.07% |
| 3 | 0.8233 | 70.99% |
| 4 | 0.7103 | 74.93% |
| 5 | 0.6100 | 78.72% |

Final Test Accuracy: **72.20%**

The trained model is saved in `models/cnn_model.pth`.

## How to Run

### 1. Build the Docker image

```bash
docker build -t apan5560-assignment2 .
```

### 2. Run the Docker container

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

## Model Training

To retrain the CNN:

```bash
uv sync
uv run python train.py
```