"""Train an unconditional GAN on all 60,000 MNIST training images."""
import argparse
import json
import random
from pathlib import Path
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from helper_lib.model import get_model, LATENT_DIM, initialize_weights
from helper_lib.trainer import train_gan

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # CPU training is portable; no CUDA/MPS is required by the API or Docker image.
    dataset = MNIST(args.data_dir, train=True, download=True)
    images = dataset.data.unsqueeze(1).float().div(127.5).sub(1)
    model = get_model("GAN")
    generator, discriminator = model.generator, model.discriminator
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_DIM)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    sample_dir = ROOT / "examples"
    sample_dir.mkdir(exist_ok=True)
    checkpoint_path = model_dir / "training_checkpoint.pt"
    history = []
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        fixed_noise = checkpoint["fixed_noise"]
        torch.set_rng_state(checkpoint["rng_state"])
    if start_epoch >= args.epochs:
        parser.error("--epochs must exceed the checkpoint epoch when resuming")
    print(f"MNIST: {len(images)} images; CPU; start epoch {start_epoch + 1}", flush=True)
    class ImageBatches:
        def __iter__(self):
            order = torch.randperm(len(images))
            for indices in order.split(args.batch_size):
                yield images[indices]

    def save_epoch(relative_epoch, losses):
        epoch = start_epoch + relative_epoch - 1
        result = {"epoch": epoch + 1, **losses}
        history.append(result)
        generator.eval()
        with torch.inference_mode():
            samples = generator(fixed_noise)
        save_image(samples, sample_dir / f"epoch_{epoch+1:02d}.png", nrow=8,
                   normalize=True, value_range=(-1, 1))
        save_image(samples, sample_dir / "generated_digits.png", nrow=8,
                   normalize=True, value_range=(-1, 1))
        torch.save(generator.state_dict(), model_dir / "generator.pt")
        torch.save({"epoch": epoch + 1, "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(), "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(), "history": history,
                    "fixed_noise": fixed_noise, "rng_state": torch.get_rng_state()}, checkpoint_path)
        metadata = {"dataset": "MNIST training split", "training_images": len(images),
                    "seed": args.seed, "batch_size": args.batch_size, "learning_rate": 0.0002,
                    "adam_betas": [0.5, 0.999], "device": "cpu", "torch_version": str(torch.__version__), "history": history,
                    "note": "GAN losses are not classification accuracy or a standalone image-quality score."}
        (model_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(result), flush=True)

    train_gan(model, ImageBatches(), criterion,
              {"generator": optimizer_g, "discriminator": optimizer_d},
              epochs=args.epochs - start_epoch, on_epoch=save_epoch)

if __name__ == "__main__":
    main()
