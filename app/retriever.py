import pickle
from pathlib import Path

import numpy as np

from app.chunker import load_documents
from app.config import (
    EMBED_MODEL_NAME,
    EMBED_MODEL_TXT,
    EMBEDDINGS_NPY,
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    RETRIEVAL_BACKEND,
    TOP_K,
    VECTORIZER_PKL,
)


def _index_missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"Индекс не найден: {path}. Запустите: uv run python scripts/build_index.py"
    )


class TfidfBackend:
    name = "tfidf"

    def __init__(self, vectorizer_path: Path, matrix_path: Path) -> None:
        import scipy.sparse

        if not vectorizer_path.exists():
            raise _index_missing(vectorizer_path)
        if not matrix_path.exists():
            raise _index_missing(matrix_path)

        with vectorizer_path.open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.matrix = scipy.sparse.load_npz(matrix_path)

    @property
    def n_rows(self) -> int:
        return self.matrix.shape[0]

    def scores(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        return cosine_similarity(query_vec, self.matrix).flatten()


class SemanticBackend:
    name = "semantic"

    def __init__(self, embeddings_path: Path, model_path: Path) -> None:
        if not embeddings_path.exists():
            raise _index_missing(embeddings_path)

        self.embeddings = np.load(embeddings_path).astype(np.float32)
        if model_path.exists():
            self.model_name = model_path.read_text(encoding="utf-8").strip()
        else:
            self.model_name = EMBED_MODEL_NAME
        self._model = None  

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def n_rows(self) -> int:
        return self.embeddings.shape[0]

    def scores(self, query: str) -> np.ndarray:
        model = self._ensure_model()
        q = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        return self.embeddings @ q


def _select_backend(backend: str):
    """Возвращает инициализированный бэкенд с откатом на TF-IDF."""
    if backend == "semantic" and EMBEDDINGS_NPY.exists():
        return SemanticBackend(EMBEDDINGS_NPY, EMBED_MODEL_TXT)
    return TfidfBackend(VECTORIZER_PKL, MATRIX_NPZ)


class Retriever:
    """Единый интерфейс поиска поверх выбранного бэкенда."""

    def __init__(
        self,
        chunks_path: Path = INDEX_CHUNKS_JSONL,
        backend: str | None = None,
        vectorizer_path: Path | None = None,
        matrix_path: Path | None = None,
    ) -> None:
        self.chunks = load_documents(chunks_path)

        if vectorizer_path is not None or matrix_path is not None:
            self.backend = TfidfBackend(
                vectorizer_path or VECTORIZER_PKL,
                matrix_path or MATRIX_NPZ,
            )
        else:
            self.backend = _select_backend((backend or RETRIEVAL_BACKEND).lower())

        if self.backend.n_rows != len(self.chunks):
            raise ValueError(
                "Число строк индекса не совпадает с числом чанков "
                f"({self.backend.n_rows} != {len(self.chunks)}). Пересоберите индекс."
            )

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if not query.strip():
            return []

        k = min(k, len(self.chunks))
        scores = self.backend.scores(query.strip())
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "name": chunk["name"],
                    "score": float(scores[idx]),
                }
            )
        return results
