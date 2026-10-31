## Assignment 3: MNIST GAN

This assignment extends the existing FastAPI application with a GAN that generates 28 × 28 grayscale handwritten digit images.

### Architecture

Generator:
- Input: a 100-dimensional noise vector.
- Linear layer: 100 → 7 × 7 × 128, followed by reshape.
- ConvTranspose2d: 128 → 64, kernel size 4, stride 2, padding 1.
- BatchNorm2d and ReLU.
- ConvTranspose2d: 64 → 1, kernel size 4, stride 2, padding 1.
- Tanh output activation.

Discriminator:
- Input: a 1 × 28 × 28 image.
- Conv2d: 1 → 64, kernel size 4, stride 2, padding 1.
- LeakyReLU with slope 0.2.
- Conv2d: 64 → 128, kernel size 4, stride 2, padding 1.
- BatchNorm2d and LeakyReLU with slope 0.2.
- Flatten and Linear layer to one output logit.


### Training

- Dataset: MNIST training set, 60,000 images.
- Pixel normalization: [-1, 1].
- Batch size: 64.
- Epochs: 20.
- Optimizer: separate Adam optimizers for both networks.
- Learning rate: 0.0002.
- Adam betas: (0.5, 0.999).
- Device used: CPU.

Install dependencies:

    uv sync

Run a short training check:

    uv run python train_gan.py --smoke

Train the full model:

    uv run python train_gan.py --epochs 20

Each invocation starts training from scratch.

The final weights are saved to `models/mnist_gan_v2.pth`. Sample grids are saved after each epoch in `samples/mnist_gan_v2/`.

### Results and Limitations

The trained generator produces multiple recognizable handwritten digit shapes, although some samples remain distorted or ambiguous. Visual quality fluctuated across epochs; epoch 20 was not consistently clearer than epoch 9.

This is an unconditional GAN: users can change the random seed, but cannot request a specific digit.

### Run the API Locally

    uv run uvicorn app.main:app --reload --port 8000

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

The existing `/embedding` and `/classify` routes remain available. The embedding route additionally requires the spaCy `en_core_web_lg` model.

### Generate Images

Endpoint: `GET /generate`

Parameters:
- `num_samples`: number of images, from 1 to 100; default 25.
- `seed`: random seed, from 0 to 4294967295; default 42.

The response is a PNG image grid.

Example requests:

http://127.0.0.1:8000/generate?num_samples=25&seed=42

http://127.0.0.1:8000/generate?num_samples=25&seed=123

Both requests were tested locally and returned different image grids. Using the same seed with the same weights and environment reproduces the same output.