# Submission

## Ссылка на репозиторий с заданием

- Repo URL: https://github.com/ManuylovPI/rag-tutorial

## Автор

- Мануйлов Павел Игоревич

## Комментарий

Реализован учебный RAG поверх отзывов **Yelp** (`Yelp/yelp_review_full`,
срез 5000 отзывов → 14975 чанков).

Полный pipeline: `prepare_datasets → ingest → chunking → index → retrieval →
demo-answer → Streamlit UI`. Ответ строится только из найденных фрагментов с
показом источников (doc_id, score, текст); при отсутствии релевантных фрагментов
система честно отказывается отвечать.

**Demo:** 3 рабочих вопроса (бары / рестораны / автосервис). Для проверки отказа использовать: quantum entanglement in particle physics

**Тесты:** 15 шт. (chunking, retrieval, eval) — все green.

**Улучшения (реализовано 2):**
1. Semantic-поиск через `sentence-transformers` (`all-MiniLM-L6-v2`) рядом с
   TF-IDF, с переключением бэкенда и автооткатом.
2. Eval-метрики `Recall@k` / `MRR@k` со сравнением бэкендов на эталонном наборе.

Инструкция запуска и логи проверок — в `README.md` и `docs/logs/`.
