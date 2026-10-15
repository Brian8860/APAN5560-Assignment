from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def get_data_loader(data_dir="data", batch_size=64, train=True):
    """
    Create a DataLoader for the CIFAR-10 dataset.

    Args:
        data_dir: Directory used to store the dataset.
        batch_size: Number of images per batch.
        train: True for training data, False for test data.

    Returns:
        PyTorch DataLoader.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.CIFAR10(
        root=str(Path(data_dir)),
        train=train,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

    return loader