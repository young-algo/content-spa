import os
import asyncio
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

async def get_embedding(text: str) -> list[float]:
    """Generates a high-quality embedding for the given text."""
    model = get_model()
    
    # Run the CPU-bound encoding in a separate thread so it doesn't block
    # the main asyncio event loop when processing multiple documents.
    embedding = await asyncio.to_thread(model.encode, text)
    return embedding.tolist()
