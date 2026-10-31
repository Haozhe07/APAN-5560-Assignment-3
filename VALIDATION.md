# Verification - 2026-10-31

- Architecture matches the specified generator and discriminator.
- Full training set: 60,000 MNIST images; ten epochs; batch size 256; CPU; seed 42.
- Actual training losses/times are recorded in models/metrics.json.
- Checkpoint restore was exercised at epoch 8; the final two epochs ran through helper_lib.trainer.train_gan.
- Both network optimizer updates and finite losses were checked.
- Local and Docker integration checks passed for PNG output, single/grid dimensions,
  seed repeatability, evaluation BatchNorm, parameter validation and missing weights.
- Existing text generation, 300-dimensional embedding and CIFAR-10 classification routes passed.
- Docker build succeeded; a real HTTP request returned the generated PNG in examples/api_samples.png.
- The generator weights inside Docker match the final local weights by SHA-256:
  ad16e108235447a77654544337b5655508a184b98e7b832ce443e8c54f327e99
- The theory PDF was rendered and both pages were visually checked.

Local demonstration (already running): http://localhost:8766/docs
Standard instructor setup: follow the Docker commands in README.md (port 8000).

To stop the local demonstration when finished:

```sh
docker stop apan5560-assignment3
docker rm apan5560-assignment3
```
