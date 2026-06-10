# План задач (tasklist)

Итерации реализации (одна итерация = один проверяемый результат):

- [x] **Iter 0 — Scaffold:** структура каталогов, `config.py`, `pyproject.toml`,
  `.gitignore`. Проверка: `import app.config`.
- [x] **Iter 1 — Data:** `prepare_datasets.py` (Yelp + синтетический фолбэк),
  5000 записей. Проверка: длина `datasets["datasets"]`.
- [x] **Iter 2 — Ingestion:** `ingest.py` → `documents.jsonl`. Проверка: число
  строк = числу записей.
- [x] **Iter 3 — Chunking:** `chunker.py` → `chunks.jsonl`. Проверка:
  `pytest tests/test_chunking.py`.
- [x] **Iter 4 — Index:** `build_index.py` (TF-IDF + semantic). Проверка: три+
  файла в `data/index/`, размер матрицы в консоли.
- [x] **Iter 5 — Retrieval:** `retriever.py`, top-k cosine. Проверка:
  `check_retrieval.py`.
- [x] **Iter 6 — Demo answer:** `generator.py` + `prompts.py`, отказ по порогу.
  Проверка: `check_generator.py` (3 demo + 1 negative).
- [x] **Iter 7 — UI:** `main.py` (Streamlit). Проверка: запуск, видны фрагменты
  с doc_id и score.
- [x] **Iter 8 — Tests & README:** 15 тестов green, воспроизводимый README.
  Проверка: `pytest tests/`.
- [x] **Улучшения:** semantic-поиск + eval-метрики (Recall@k, MRR@k).

## Критерий готовности

Все пункты отмечены, проверки проходят, README позволяет запуск с нуля.
