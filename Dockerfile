FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
WORKDIR /code
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
RUN uv run --frozen python -m spacy download en_core_web_lg
COPY app ./app
COPY helper_lib ./helper_lib
COPY models/generator.pt models/cnn_cifar10.pt ./models/
EXPOSE 80
CMD ["uv", "run", "--frozen", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]
