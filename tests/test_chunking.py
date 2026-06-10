"""Тесты нарезки текста на чанки."""

from app.chunker import chunk_document, chunk_text, run


def test_chunk_text_respects_max_size():
    text = "Первый абзац.\n\n" + "word " * 200
    chunks = chunk_text(text, max_chars=400, overlap=50)
    assert chunks
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_splits_by_paragraphs():
    text = "Отзыв про кофейню.\n\nОтзыв про ресторан."
    chunks = chunk_text(text, max_chars=400, overlap=50)
    assert len(chunks) == 1
    assert "кофейню" in chunks[0]
    assert "ресторан" in chunks[0]


def test_chunk_text_overlap_between_chunks():
    para1 = "A" * 300
    para2 = "B" * 300
    text = f"{para1}\n\n{para2}"
    chunks = chunk_text(text, max_chars=400, overlap=50)
    assert len(chunks) >= 2
    assert chunks[1].startswith(chunks[0][-50:])


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_document_has_doc_id():
    doc = {
        "doc_id": "42",
        "name": "5/5 · Кафе и напитки · отличный латте",
        "text": "Рейтинг: 5/5 звёзд. Категория: Кафе и напитки.\n\nGreat latte.",
    }
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    assert chunks[0]["doc_id"] == "42"
    assert chunks[0]["chunk_id"] == "42_0"
    assert chunks[0]["name"].startswith("5/5")


def test_run_creates_chunks_jsonl(tmp_path):
    docs = tmp_path / "documents.jsonl"
    docs.write_text(
        '{"doc_id": "0", "name": "A", "text": "Короткий отзыв."}\n',
        encoding="utf-8",
    )
    out = tmp_path / "chunks.jsonl"
    count = run(input_path=docs, output_path=out)
    assert count == 1
    assert out.exists()
    line = out.read_text(encoding="utf-8").strip()
    assert '"doc_id": "0"' in line
