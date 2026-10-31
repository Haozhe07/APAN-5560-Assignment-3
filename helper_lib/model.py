"""Exact Assignment 3 GAN architecture (NCHW image layout)."""
import torch
from torch import nn

LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 7 * 7 * 128)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.layers(self.fc(noise).reshape(-1, 128, 7, 7))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.layers(images)

def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class GAN(nn.Module):
    """Bundle both networks for the helper-library factory and trainer."""
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name):
    if model_name.upper() == "GAN":
        return GAN()
    raise ValueError("This assignment implements GAN; request get_model('GAN')")
