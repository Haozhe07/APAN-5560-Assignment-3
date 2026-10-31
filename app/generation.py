from functools import lru_cache
from pathlib import Path
import torch
from helper_lib.model import Generator
from helper_lib.generator import generate_samples

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "generator.pt"

@lru_cache(maxsize=1)
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model

def generate_image(count=1, seed=None):
    return generate_samples(load_generator(), device="cpu", num_samples=count, seed=seed)
