"""Shared preprocessing and trained CIFAR-10 inference."""
from pathlib import Path
from functools import lru_cache
import torch
from torchvision import transforms
from app.cnn_model import CNNModel

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cnn_cifar10.pt"
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@lru_cache(maxsize=1)
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model

def classify(image):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(load_model()(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    return {"class_id": index, "class": CLASSES[index], "confidence": float(probabilities[index]),
            "probabilities": dict(zip(CLASSES, probabilities.tolist()))}
