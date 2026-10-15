from io import BytesIO
from pathlib import Path

import spacy
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

from helper_lib.data_loader import CIFAR10_CLASSES
from helper_lib.model import get_model
from helper_lib.utils import load_model


app = FastAPI()

# Assignment 1: Word embedding
nlp = spacy.load("en_core_web_lg")


# Assignment 2: CNN image classification
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_model.pth"

cnn_model = get_model("CNN")
cnn_model = load_model(cnn_model, filepath=MODEL_PATH, device="cpu")

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


class EmbeddingRequest(BaseModel):
    word: str


def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


@app.get("/")
def read_root():
    return {
        "message": "FastAPI word embedding and CNN classification API is running"
    }


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)

    return {
        "word": request.word,
        "embedding": embedding.tolist()
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded image using the CIFAR-10 CNN."""

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )
    finally:
        await file.close()

    input_tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():
        outputs = cnn_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_index = predicted_class.item()

    return {
        "filename": file.filename,
        "predicted_class": CIFAR10_CLASSES[class_index],
        "class_index": class_index,
        "confidence": round(confidence.item(), 4)
    }