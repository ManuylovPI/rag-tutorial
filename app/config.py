

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INDEX = ROOT / "data" / "index"

RAW_DATASETS = DATA_RAW / "datasets.json"
DOCUMENTS_JSONL = DATA_PROCESSED / "documents.jsonl"
CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"

VECTORIZER_PKL = DATA_INDEX / "vectorizer.pkl"
MATRIX_NPZ = DATA_INDEX / "matrix.npz"

EMBEDDINGS_NPY = DATA_INDEX / "embeddings.npy"
EMBED_MODEL_TXT = DATA_INDEX / "embed_model.txt"

INDEX_CHUNKS_JSONL = DATA_INDEX / "chunks.jsonl"

TOP_K = 3
CHUNK_MAX_CHARS = 400
CHUNK_OVERLAP = 50

RETRIEVAL_BACKEND = os.environ.get("RAG_BACKEND", "semantic").strip().lower()

EMBED_MODEL_NAME = os.environ.get(
    "RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
