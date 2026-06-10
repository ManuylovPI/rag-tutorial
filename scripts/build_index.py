import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.chunker import load_documents, run as chunk_run
from app.config import (
    CHUNKS_JSONL,
    DATA_INDEX,
    EMBED_MODEL_NAME,
    EMBED_MODEL_TXT,
    EMBEDDINGS_NPY,
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    VECTORIZER_PKL,
)
from ingest import run as ingest_run


def build_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, scipy.sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_tfidf(
    vectorizer: TfidfVectorizer,
    matrix: scipy.sparse.csr_matrix,
) -> None:
    with VECTORIZER_PKL.open("wb") as f:
        pickle.dump(vectorizer, f)
    scipy.sparse.save_npz(MATRIX_NPZ, matrix)


def build_semantic(texts: list[str]) -> bool:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[i] sentence-transformers не установлен — semantic-индекс пропущен. "
            "Система будет работать на TF-IDF."
        )
        return False

    try:
        print(f"[i] Кодирование {len(texts)} чанков моделью {EMBED_MODEL_NAME}…")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  
            convert_to_numpy=True,
        ).astype(np.float32)
    except Exception as exc:  
        print(
            f"[!] Не удалось построить semantic-индекс: {exc}\n"
            "[!] Модель эмбеддингов недоступна (нет интернета или модель не "
            "скачана). Semantic-индекс пропущен — система работает на TF-IDF."
        )
        return False

    np.save(EMBEDDINGS_NPY, embeddings)
    EMBED_MODEL_TXT.write_text(EMBED_MODEL_NAME, encoding="utf-8")
    print(f"[i] Semantic-индекс сохранён: {embeddings.shape}")
    return True


def run() -> int:
    DATA_INDEX.mkdir(parents=True, exist_ok=True)

    doc_count = ingest_run()
    chunk_count = chunk_run()
    chunks = load_documents(CHUNKS_JSONL)
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("Нет чанков для индексации")

    vectorizer, matrix = build_tfidf(texts)
    save_tfidf(vectorizer, matrix)

    semantic_ok = build_semantic(texts)

    shutil.copy2(CHUNKS_JSONL, INDEX_CHUNKS_JSONL)

    backends = "TF-IDF + semantic" if semantic_ok else "TF-IDF"
    print(
        f"Документов: {doc_count}, чанков: {chunk_count}, "
        f"TF-IDF матрица: {matrix.shape}, бэкенды: {backends}"
    )
    return chunk_count


def main() -> None:
    run()
    print(f"Индекс сохранён -> {DATA_INDEX}")


if __name__ == "__main__":
    main()
