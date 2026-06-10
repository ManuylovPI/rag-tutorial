"""Дополнительный (собственный) тест: корректность eval-метрик retrieval@k.

Проверяет логику метрик Recall@k и MRR@k на контролируемом мини-индексе с
заранее известными правильными ответами. Это наш вклад сверх базовых тестов
chunking/retrieval — он защищает метрики качества (улучшение №7) от регрессий.
"""

import json
import pickle
from pathlib import Path

import pytest
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retriever import Retriever
from scripts.eval import category_matches, evaluate


def test_category_matches_is_case_insensitive_substring():
    assert category_matches("5/5 · Бары и ночная жизнь · ...", "Бары")
    assert category_matches("1/5 · АВТО и сервис", "авто")
    assert not category_matches("5/5 · Рестораны и еда", "Бары")


@pytest.fixture
def labelled_index(tmp_path: Path) -> dict[str, Path]:
    """Индекс из двух чётко различимых категорий с предсказуемым поиском."""
    chunks = [
        {
            "chunk_id": "0_0",
            "doc_id": "0",
            "name": "1/5 · Бары и ночная жизнь · бармен",
            "text": "Категория: Бары. rude bartender cocktails drinks pub beer",
        },
        {
            "chunk_id": "1_0",
            "doc_id": "1",
            "name": "5/5 · Авто и сервис · механик",
            "text": "Категория: Авто. mechanic oil change tires brakes repair car",
        },
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = [c["text"] for c in chunks]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts)

    vec_path = tmp_path / "vectorizer.pkl"
    mat_path = tmp_path / "matrix.npz"
    with vec_path.open("wb") as f:
        pickle.dump(vec, f)
    scipy.sparse.save_npz(mat_path, matrix)

    return {
        "chunks_path": chunks_path,
        "vectorizer_path": vec_path,
        "matrix_path": mat_path,
    }


def test_evaluate_perfect_recall_and_mrr(labelled_index, monkeypatch):
    """Если каждый вопрос точно соответствует своей категории → метрики = 1.0."""
    import scripts.eval as eval_mod

    gold = [
        ("rude bartender cocktails", "Бары"),
        ("mechanic oil change brakes", "Авто"),
    ]
    monkeypatch.setattr(eval_mod, "GOLD", gold)

    r = Retriever(**labelled_index)
    recall, mrr = evaluate(r, k=2)
    assert recall == 1.0
    assert mrr == 1.0


def test_evaluate_counts_miss(labelled_index, monkeypatch):
    """Вопрос без релевантной категории в корпусе снижает recall."""
    import scripts.eval as eval_mod

    gold = [
        ("rude bartender cocktails", "Бары"),
        ("spotless hotel room reception", "Отели"),
    ]
    monkeypatch.setattr(eval_mod, "GOLD", gold)

    r = Retriever(**labelled_index)
    recall, mrr = evaluate(r, k=2)
    assert recall == 0.5  
    assert 0.0 < mrr <= 0.5
