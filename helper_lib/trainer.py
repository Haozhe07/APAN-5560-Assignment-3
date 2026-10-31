"""Alternating discriminator/generator updates used by train.py."""
import time
import torch
from .model import LATENT_DIM


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10, on_epoch=None):
    """optimizer maps generator/discriminator to their separate optimizers.

    data_loader yields image tensors or (images, labels); labels are ignored.
    Images must be normalized to [-1, 1]. on_epoch receives (epoch, losses).
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(device)
    generator, discriminator = model.generator, model.discriminator
    optimizer_g, optimizer_d = optimizer["generator"], optimizer["discriminator"]
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        sum_g = sum_d = seen = 0
        for batch in data_loader:
            real = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            size = len(real)
            real_targets = torch.ones(size, 1, device=device)
            fake_targets = torch.zeros(size, 1, device=device)
            optimizer_d.zero_grad(set_to_none=True)
            fake = generator(torch.randn(size, LATENT_DIM, device=device))
            loss_d = (criterion(discriminator(real), real_targets) +
                      criterion(discriminator(fake.detach()), fake_targets)) / 2
            loss_d.backward()
            optimizer_d.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            try:
                loss_g = criterion(discriminator(fake), real_targets)
                loss_g.backward()
                optimizer_g.step()
            finally:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            sum_g += loss_g.item() * size
            sum_d += loss_d.item() * size
            seen += size
        if seen == 0:
            raise ValueError("Training loader must not be empty")
        result = {"generator_loss": sum_g / seen, "discriminator_loss": sum_d / seen,
                  "seconds": round(time.monotonic() - started, 2)}
        if on_epoch is not None:
            on_epoch(epoch + 1, result)
    return model
