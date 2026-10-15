import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import save_model


def main():
    torch.manual_seed(42)

    # Use Apple GPU if available
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load CIFAR-10 training and test data
    train_loader = get_data_loader(
        data_dir="data",
        batch_size=64,
        train=True
    )

    test_loader = get_data_loader(
        data_dir="data",
        batch_size=64,
        train=False
    )

    model = get_model("CNN")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the CNN
    model = train_model(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5
    )

    # Evaluate on the test dataset
    evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Save the trained weights
    save_model(model, filepath="models/cnn_model.pth")


if __name__ == "__main__":
    main()