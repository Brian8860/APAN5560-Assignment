from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def generate_samples(
    model,
    device="cpu",
    num_samples=10,
    noise=None,
    save_path=None,
    show=True
):
    """Generate images, optionally save a grid, and display it."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    generator = model.generator
    was_training = generator.training

    generator.to(device)
    generator.eval()

    try:
        if noise is None:
            noise = torch.randn(num_samples, 100, device=device)
        else:
            noise = noise.to(device)

        with torch.inference_mode():
            images = generator(noise)

            # Convert from [-1, 1] to [0, 1].
            images = ((images + 1) / 2).clamp(0, 1).cpu()
    finally:
        generator.train(was_training)

    nrow = min(5, images.size(0))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(images, save_path, nrow=nrow)

    if show:
        import matplotlib.pyplot as plt

        grid = make_grid(images, nrow=nrow)

        plt.figure(figsize=(10, 4))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return images