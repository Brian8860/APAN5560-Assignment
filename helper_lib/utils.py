from pathlib import Path

import torch


def save_model(model, filepath="models/cnn_model.pth"):
    """Save the trained model weights."""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), filepath)

    print(f"Model saved to: {filepath}")


def load_model(model, filepath="models/cnn_model.pth", device="cpu"):
    """Load trained model weights."""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model weights not found: {filepath}")

    state_dict = torch.load(
        filepath,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model