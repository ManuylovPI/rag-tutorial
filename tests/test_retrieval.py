"""Тесты retrieval и demo-ответа на изолированном мини-индексе (TF-IDF)."""

import json
import pickle
from pathlib import Path

import pytest
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import TOP_K
from app.generator import ask
from app.prompts import REFUSAL_NO_CONTEXT
from app.retriever import Retriever


@pytest.fixture
def mini_index(tmp_path: Path) -> dict[str, Path]:
    """Мини-индекс из трёх отзывов для изолированных тестов retrieval."""
    chunks = [
        {
            "chunk_id": "0_0",
            "doc_id": "0",
            "name": "1/5 · Бары и ночная жизнь · грубый бармен",
            "text": "Категория: Бары и ночная жизнь. The bartender was rude and ignored us.",
        },
        {
            "chunk_id": "1_0",
            "doc_id": "1",
            "name": "5/5 · Авто и сервис · честный механик",
            "text": "Категория: Авто и сервис. Fast oil change and an honest mechanic.",
        },
        {
            "chunk_id": "2_0",
            "doc_id": "2",
            "name": "5/5 · Рестораны и еда · вкусная паста",
            "text": "Категория: Рестораны и еда. Delicious pasta and a great waiter.",
        },
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    vectorizer_path = tmp_path / "vectorizer.pkl"
    matrix_path = tmp_path / "matrix.npz"
    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    scipy.sparse.save_npz(matrix_path, matrix)

    return {
        "chunks_path": chunks_path,
        "vectorizer_path": vectorizer_path,
        "matrix_path": matrix_path,
    }


def test_search_returns_k_results(mini_index):
    r = Retriever(**mini_index)
    results = r.search("bartender rude", k=3)
    assert len(results) == 3


def test_search_results_have_required_fields(mini_index):
    r = Retriever(**mini_index)
    results = r.search("mechanic oil change", k=TOP_K)
    assert results
    for hit in results:
        assert "doc_id" in hit
        assert "text" in hit
        assert "score" in hit
        assert "name" in hit
        assert isinstance(hit["score"], float)


def test_search_prefers_matching_category(mini_index):
    r = Retriever(**mini_index)
    results = r.search("honest mechanic fast oil change", k=1)
    assert results[0]["doc_id"] == "1"
    assert results[0]["score"] > 0


def test_search_empty_query_returns_empty(mini_index):
    r = Retriever(**mini_index)
    assert r.search("") == []
    assert r.search("   ") == []


def test_ask_sources_contain_doc_id(mini_index):
    result = ask("bartender rude ignored", retriever=Retriever(**mini_index))
    assert result["sources"]
    assert all("doc_id" in src for src in result["sources"])
    assert result["sources"][0]["doc_id"] == "0"


def test_ask_refuses_without_relevant_context(mini_index):
    result = ask(
        "quantum entanglement orbital mechanics rocket",
        retriever=Retriever(**mini_index),
    )
    assert result["answer"] == REFUSAL_NO_CONTEXT
