from functools import lru_cache
from app.embedding_model import EmbeddingModel

@lru_cache(maxsize=1)
def load_embedding():
    return EmbeddingModel()
