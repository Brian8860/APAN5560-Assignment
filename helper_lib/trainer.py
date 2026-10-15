import torch


def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    """
    Train a PyTorch neural network.

    Args:
        model: Neural network to train.
        data_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU.
        epochs: Number of training epochs.

    Returns:
        Trained model.
    """

    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Track training metrics
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )

    return model