import torch


def evaluate_model(
    model,
    data_loader,
    criterion,
    device="cpu"
):
    """
    Evaluate a PyTorch neural network.

    Args:
        model: Trained neural network.
        data_loader: Test DataLoader.
        criterion: Loss function.
        device: CPU or GPU.

    Returns:
        avg_loss: Average loss on the test dataset.
        accuracy: Classification accuracy (%).
    """

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    print(
        f"Test Loss: {avg_loss:.4f} "
        f"Test Accuracy: {accuracy:.2f}%"
    )

    return avg_loss, accuracy