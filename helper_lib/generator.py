"""Generate MNIST samples and return a displayable/savable image grid."""
import torch
from PIL import Image
from torchvision.utils import make_grid
from .model import GAN, LATENT_DIM


def generate_samples(model, device="cpu", num_samples=10, seed=None):
    if not 1 <= num_samples <= 64:
        raise ValueError("num_samples must be between 1 and 64")
    generator = model.generator if isinstance(model, GAN) else model
    generator = generator.to(device)
    rng = torch.Generator(device=device)
    rng.seed() if seed is None else rng.manual_seed(seed)
    noise = torch.randn(num_samples, LATENT_DIM, generator=rng, device=device)
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            images = generator(noise).add(1).div(2).clamp(0, 1).cpu()
    finally:
        generator.train(was_training)
    if num_samples == 1:
        pixels = images[0, 0].mul(255).round().byte().numpy()
    else:
        grid = make_grid(images, nrow=min(8, num_samples), padding=2, pad_value=0)
        pixels = grid[0].mul(255).round().byte().numpy()
    return Image.fromarray(pixels)
