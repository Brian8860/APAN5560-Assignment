from functools import lru_cache
from io import BytesIO
from pathlib import Path

import spacy
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms
from torchvision.utils import make_grid

from helper_lib.data_loader import CIFAR10_CLASSES
from helper_lib.model import get_model
from helper_lib.utils import load_model


ROOT = Path(__file__).resolve().parent.parent

CNN_MODEL_PATH = ROOT / "models" / "cnn_model.pth"
GAN_MODEL_PATH = ROOT / "models" / "mnist_gan_v2.pth"

app = FastAPI(
    title="APAN 5560 API",
    description="Word embeddings, CNN classification, and GAN generation."
)


# Load models only when their endpoint is first used.
@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("en_core_web_lg")


@lru_cache(maxsize=1)
def get_cnn():
    model = get_model("CNN")
    return load_model(
        model,
        filepath=CNN_MODEL_PATH,
        device="cpu"
    )


@lru_cache(maxsize=1)
def get_gan():
    model = get_model("GAN")
    return load_model(
        model,
        filepath=GAN_MODEL_PATH,
        device="cpu"
    )


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


@app.get("/")
def read_root():
    return {
        "message": "APAN 5560 API is running",
        "endpoints": [
            "/embedding",
            "/classify",
            "/generate"
        ]
    }


# Assignment 1: Word embeddings
@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    try:
        nlp = get_nlp()
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="The spaCy en_core_web_lg model is not installed."
        ) from exc

    embedding = nlp(request.word).vector

    return {
        "word": request.word,
        "embedding": embedding.tolist()
    }


# Assignment 2: CNN image classification
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        if file.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported."
            )

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        ) from exc

    finally:
        await file.close()

    try:
        model = get_cnn()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="CNN weights are missing: models/cnn_model.pth"
        ) from exc

    input_tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    class_index = predicted_class.item()

    return {
        "filename": file.filename,
        "predicted_class": CIFAR10_CLASSES[class_index],
        "class_index": class_index,
        "confidence": round(confidence.item(), 4)
    }


# Assignment 3: GAN image generation
@app.get(
    "/generate",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "A grid of generated handwritten digits."
        }
    },
    response_class=Response
)
def generate_digits(
    num_samples: int = Query(default=25, ge=1, le=100),
    seed: int = Query(default=42, ge=0, le=4294967295)
):
    try:
        model = get_gan()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="GAN weights are missing: models/mnist_gan_v2.pth"
        ) from exc

    # A local random generator makes results repeatable without
    # changing the application's global random seed.
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(seed)

    noise = torch.randn(
        num_samples,
        100,
        generator=random_generator
    )

    with torch.inference_mode():
        images = model.generator(noise)
        images = ((images + 1) / 2).clamp(0, 1)

        grid = make_grid(
            images,
            nrow=min(5, num_samples),
            padding=2
        )

        pixels = (
            grid.mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

    image = Image.fromarray(pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="generated_digits.png"'
        }
    )