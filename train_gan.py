import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from helper_lib.data_loader import get_mnist_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import save_model


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu"
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Test training with 256 images for one epoch."
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple GPU (MPS) is unavailable.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")

    torch.manual_seed(42)
    device = args.device

    loader = get_mnist_loader(
        data_dir=str(ROOT / "data"),
        batch_size=args.batch_size,
        train=True
    )

    if args.smoke:
        loader = DataLoader(
            Subset(loader.dataset, range(256)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        epochs = 1
        run_name = "smoke_v2"
    else:
        epochs = args.epochs
        run_name = "mnist_gan_v2"

    model = get_model("GAN").to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizers = {
        "generator": torch.optim.Adam(
            model.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        ),
        "discriminator": torch.optim.Adam(
            model.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
    }

    fixed_noise = torch.randn(25, 100)

    sample_dir = ROOT / "samples" / run_name
    model_path = ROOT / "models" / f"{run_name}.pth"

    print(f"Device: {device}", flush=True)
    print(f"Training images: {len(loader.dataset)}", flush=True)
    print(f"Epochs: {epochs}", flush=True)

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===", flush=True)

        model = train_gan(
            model=model,
            data_loader=loader,
            criterion=criterion,
            optimizer=optimizers,
            device=device,
            epochs=1
        )

        image_path = sample_dir / f"epoch_{epoch:03d}.png"

        generate_samples(
            model=model,
            device=device,
            num_samples=25,
            noise=fixed_noise,
            save_path=image_path,
            show=False
        )

        save_model(model, filepath=model_path)

        print(
            f"Sample images saved: {image_path}",
            flush=True
        )

    print("\nTraining finished.", flush=True)
    print(f"Model weights: {model_path}", flush=True)


if __name__ == "__main__":
    main()