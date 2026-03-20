"""Shared sentence embedding model loader."""

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = None

    def get_model() -> SentenceTransformer:
        global _MODEL
        if _MODEL is None:
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

    def get_model():
        raise ImportError("sentence-transformers is required for embedding analysis")
