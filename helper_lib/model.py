import torch
import torch.nn as nn
import torch.nn.functional as F


# Assignment 2: CNN image classifier
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Assignment 3: Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(100, 7 * 7 * 128)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(noise.size(0), 128, 7, 7)
        return self.layers(x)


# Assignment 3: Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, images):
        # Raw logits for BCEWithLogitsLoss.
        # Apply torch.sigmoid() only when probabilities are needed.
        return self.layers(images)


class GAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name="CNN"):
    if model_name == "CNN":
        return CNN()

    if model_name == "GAN":
        return GAN()

    raise ValueError(f"Unsupported model: {model_name}")