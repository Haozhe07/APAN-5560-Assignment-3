"""FastAPI image-generation endpoint; original text APIs are preserved."""
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query, Response, File, UploadFile
from pydantic import BaseModel, Field
from app.bigram_model import BigramModel

app = FastAPI(title="APAN 5560 Assignment 3", description="MNIST GAN image generation")
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(default=20, ge=1, le=1000)

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def root():
    return {"message": "Assignment 3 API", "image_endpoint": "/generate-image"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    return {"generated_text": bigram_model.generate_text(request.start_word, request.length)}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    from app.legacy_embedding import load_embedding
    return {"word": request.word, "embedding": load_embedding().calculate_embedding(request.word)}

@app.get("/generate-image", responses={200: {"content": {"image/png": {}}}},
         response_class=Response)
def generate_image(count: int = Query(default=1, ge=1, le=64),
                   seed: int | None = Query(default=None, ge=0, le=2**63-1)):
    """Return a PNG digit or grid; count 1-64; optional seed repeats the same noise.

    This is an unconditional GAN. It cannot request a particular digit label.
    """
    from app.generation import generate_image as generate
    try:
        image = generate(count=count, seed=seed)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Trained weights missing; run python train.py")
    output = BytesIO()
    image.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png",
                    headers={"Content-Disposition": 'inline; filename="generated_digits.png"'})

@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    """Upload a JPEG or PNG; return a CIFAR-10 class and probabilities."""
    from io import BytesIO
    from PIL import Image, UnidentifiedImageError
    from app.image_classifier import classify
    data = file.file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be at most 10 MiB")
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return classify(image)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Trained model missing; run python train.py")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Upload a valid image")
