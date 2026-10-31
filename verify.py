"""Meaningful architecture, training-update and HTTP contract checks."""
import io
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
from fastapi.testclient import TestClient
from helper_lib.model import get_model, Generator, Discriminator
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from app.main import app

torch.set_num_threads(4)
torch.manual_seed(123)
model = get_model("GAN")
noise = torch.randn(4, 100)
images = model(noise)
assert images.shape == (4, 1, 28, 28)
assert images.min() >= -1 and images.max() <= 1
probabilities = model.discriminator(images)
assert probabilities.shape == (4, 1)
assert ((probabilities >= 0) & (probabilities <= 1)).all()
assert model.generator.fc.out_features == 7*7*128
convolutions = [m for m in model.generator.modules() if isinstance(m, nn.ConvTranspose2d)]
assert [(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding) for m in convolutions] == [
    (128,64,(4,4),(2,2),(1,1)), (64,1,(4,4),(2,2),(1,1))]
convolutions = [m for m in model.discriminator.modules() if isinstance(m, nn.Conv2d)]
assert [(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding) for m in convolutions] == [
    (1,64,(4,4),(2,2),(1,1)), (64,128,(4,4),(2,2),(1,1))]
old_g = model.generator.fc.weight.detach().clone()
old_d = model.discriminator.layers[0].weight.detach().clone()
losses = []
optimizers = {"generator": torch.optim.Adam(model.generator.parameters(), lr=0.0002),
              "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=0.0002)}
train_gan(model, [torch.rand(4,1,28,28)*2-1], nn.BCELoss(), optimizers,
          epochs=1, on_epoch=lambda epoch, value: losses.append(value))
assert torch.isfinite(torch.tensor([losses[0]["generator_loss"], losses[0]["discriminator_loss"]])).all()
assert not torch.equal(old_g, model.generator.fc.weight)
assert not torch.equal(old_d, model.discriminator.layers[0].weight)
model.eval()
bn = next(m for m in model.generator.modules() if isinstance(m, nn.BatchNorm2d))
bn_mean = bn.running_mean.clone()
assert generate_samples(model, num_samples=4, seed=42).size == (122,32)
assert torch.equal(bn_mean, bn.running_mean)
client = TestClient(app)
assert client.get('/').status_code == 200
first = client.get('/generate-image?count=1&seed=42')
assert first.status_code == 200 and first.headers['content-type'] == 'image/png'
assert Image.open(io.BytesIO(first.content)).size == (28,28)
assert first.content == client.get('/generate-image?count=1&seed=42').content
assert first.content != client.get('/generate-image?count=1&seed=43').content
batch = client.get('/generate-image?count=16&seed=42')
assert batch.status_code == 200
assert Image.open(io.BytesIO(batch.content)).size == (242,62)
for query in ['count=0','count=65','seed=-1','seed=9223372036854775808','count=abc']:
    assert client.get('/generate-image?'+query).status_code == 422
with patch('app.generation.generate_image', side_effect=FileNotFoundError):
    assert client.get('/generate-image').status_code == 503
assert client.post('/generate', json={'start_word':'the','length':5}).status_code == 200
assert len(client.post('/embedding', json={'word':'cat'}).json()['embedding']) == 300
image = io.BytesIO(); Image.new('RGB',(64,64)).save(image,format='PNG')
assert client.post('/classify', files={'file':('test.png',image.getvalue(),'image/png')}).status_code == 200
print('PASS: specified architecture, both optimizer updates, finite losses, eval BatchNorm, trained PNG inference, seeds, image grids, request validation, missing weights, and legacy text/embedding/CNN endpoints')
