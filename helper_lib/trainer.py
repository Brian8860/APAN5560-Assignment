import torch


# Assignment 2: CNN training
def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if total == 0:
            raise ValueError("The training data loader is empty.")

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {running_loss / total:.4f} "
            f"Accuracy: {100 * correct / total:.2f}%",
            flush=True
        )

    return model


# Assignment 3: GAN training
def train_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    """
    Use BCEWithLogitsLoss with the discriminator's raw logits.

    optimizer is a dictionary with:
        "generator": generator optimizer
        "discriminator": discriminator optimizer

    Move the model to device before constructing the optimizers.
    """
    model = model.to(device)

    generator = model.generator
    discriminator = model.discriminator

    optimizer_g = optimizer["generator"]
    optimizer_d = optimizer["discriminator"]

    for epoch in range(epochs):
        model.train()
        discriminator.requires_grad_(True)

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_images = 0

        for batch_index, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # 1. Update the discriminator.
            optimizer_d.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images.detach())

            real_loss = criterion(real_logits, real_targets)
            fake_loss = criterion(fake_logits, fake_targets)
            d_loss = (real_loss + fake_loss) / 2

            if not torch.isfinite(d_loss).item():
                raise RuntimeError(
                    "Discriminator loss became non-finite."
                )

            d_loss.backward()
            optimizer_d.step()

            # 2. Update the generator.
            # Keep the discriminator in training mode so BatchNorm
            # behaves consistently across both training phases.
            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)

            try:
                fake_logits = discriminator(fake_images)
                g_loss = criterion(fake_logits, real_targets)

                if not torch.isfinite(g_loss).item():
                    raise RuntimeError(
                        "Generator loss became non-finite."
                    )

                g_loss.backward()
                optimizer_g.step()
            finally:
                discriminator.requires_grad_(True)

            total_d_loss += d_loss.item() * batch_size
            total_g_loss += g_loss.item() * batch_size
            total_images += batch_size

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Batch [{batch_index + 1}/{len(data_loader)}] "
                    f"D Loss: {d_loss.item():.4f} "
                    f"G Loss: {g_loss.item():.4f}",
                    flush=True
                )

        if total_images == 0:
            raise ValueError("The training data loader is empty.")

        print(
            f"Epoch summary | "
            f"D Loss: {total_d_loss / total_images:.4f} | "
            f"G Loss: {total_g_loss / total_images:.4f}",
            flush=True
        )

    return model