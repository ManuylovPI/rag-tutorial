# Логи проверок (TF-IDF backend, чистая сборка)

> Примечание: на машине с доступом к Hugging Face сборка также строит
> semantic-индекс, и eval.py показывает сравнение TF-IDF vs semantic.

## build_index.py
```
[!] Модель эмбеддингов недоступна (нет интернета или модель не скачана). Semantic-индекс пропущен — система работает на TF-IDF.
Документов: 5000, чанков: 6316, TF-IDF матрица: (6316, 324), бэкенды: TF-IDF
Индекс сохранён -> /home/claude/yelp-rag/data/index
```

## check_retrieval.py
```
=== Проверка Retriever (итерация 5) ===

OK: индекс загружен, активный бэкенд = tfidf
Чанков в индексе: 6316

Запрос: «rude bartender ignored us at the bar»
Ожидание: слова есть в данных -> score > 0, категория «Бары»
Получено результатов: 3
  [1] doc_id=3203, score=0.4949
      Рейтинг: 1/5 звёзд. Категория: Бары и ночная жизнь.  Such a letdown. Unfortunately, the bartender ignored us for half an…
  [2] doc_id=930, score=0.4949
      Рейтинг: 1/5 звёзд. Категория: Бары и ночная жизнь.  Such a letdown. Unfortunately, the bartender ignored us for half an…
  [3] doc_id=4995, score=0.4714
      Рейтинг: 1/5 звёзд. Категория: Бары и ночная жизнь.  I won't be returning. Unfortunately, the bartender ignored us for h…

Запрос: «delicious pizza and friendly waiter»
Ожидание: слова есть в данных -> score > 0, категория «Рестораны»
Получено результатов: 3
  [1] doc_id=1829, score=0.2415
      Рейтинг: 5/5 звёзд. Категория: Рестораны и еда.  I keep coming back here. I really enjoyed the pasta and the wood-fired …
  [2] doc_id=3505, score=0.2415
      Рейтинг: 5/5 звёзд. Категория: Рестораны и еда.  I keep coming back here. I really enjoyed the pasta and the wood-fired …
  [3] doc_id=2604, score=0.2415
      Рейтинг: 4/5 звёзд. Категория: Рестораны и еда.  I keep coming back here. I really enjoyed the pasta and the wood-fired …

Запрос: «how to configure a wifi router at home»
Ожидание: слов нет в отзывах -> низкий score / отказ на стороне генератора
Получено результатов: 3
  [1] doc_id=4999, score=0.0000
      Рейтинг: 2/5 звёзд. Категория: Бары и ночная жизнь.  Such a letdown. Unfortunately, the bartender ignored us for half an…
  [2] doc_id=4998, score=0.0000
      Рейтинг: 1/5 звёзд. Категория: Бары и ночная жизнь.  Worst experience I've had in a while. Unfortunately, watered-down d…
  [3] doc_id=4997, score=0.0000
      Рейтинг: 4/5 звёзд. Категория: Авто и сервис.Hands down one of my favorites. I really enjoyed a well-stocked boutique wi…

=== Итог ===
Если для каждого запроса видно 3 результата с полями doc_id / score / text — итерация 5 работает.
```

## check_generator.py
```

--- Demo 1 (бары): «rude bartender and watered down drinks» ---
Ответ:
На основании найденных отзывов:

[1] 2/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, watered-down drinks and stick…
doc_id=4413, score=0.42
Рейтинг: 2/5 звёзд. Категория: Бары и ночная жизнь.

Such a letdown. Unfortunately, watered-down drinks and sticky tables. Never again.

[2] 2/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, watered-down drinks and stick…
doc_id=3967, score=0.42
Рейтинг: 2/5 звёзд. Категория: Бары и ночная жизнь.

Such a letdown. Unfortunately, watered-down drinks and sticky tables. Never again.

[3] 1/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, watered-down drinks and stick…
doc_id=4523, score=0.42
Рейтинг: 1/5 звёзд. Категория: Бары и ночная жизнь.

Such a letdown. Unfortunately, watered-down drinks and sticky tables. Never again.

Источников: 3
  [1] doc_id=4413, score=0.4181, name=2/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, w…
  [2] doc_id=3967, score=0.4181, name=2/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, w…
  [3] doc_id=4523, score=0.4181, name=1/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, w…

--- Demo 2 (рестораны): «delicious food and great service» ---
Ответ:
На основании найденных отзывов:

[1] 5/5 · Авто и сервис · I keep coming back here. I really enjoyed a well-stocked bou…
doc_id=4990, score=0.26
Рейтинг: 5/5 звёзд. Категория: Авто и сервис.

I keep coming back here. I really enjoyed a well-stocked boutique with great deals. The service overall set the tone for the whole visit, and the atmosphere played a big part in how I felt by the end. I came in with certain expectations and left thinking carefully about whether I would tell my friends to try it for themselves. Worth every penny.

[2] 5/5 · Авто и сервис · I keep coming back here. I really enjoyed a well-stocked bou…
doc_id=3406, score=0.26
Рейтинг: 5/5 звёзд. Категория: Авто и сервис.

I keep coming back here. I really enjoyed a well-stocked boutique with great deals. The service overall set the tone for the whole visit, and the atmosphere played a big part in how I felt by the end. I came in with certain expectations and left thinking carefully about whether I would tell my friends to try it for themselves. Worth every penny.

[3] 5/5 · Авто и сервис · I keep coming back here. I really enjoyed a well-stocked bou…
doc_id=2366, score=0.26
Рейтинг: 5/5 звёзд. Категория: Авто и сервис.

I keep coming back here. I really enjoyed a well-stocked boutique with great deals. The service overall set the tone for the whole visit, and the atmosphere played a big part in how I felt by the end. I came in with certain expectations and left thinking carefully about whether I would tell my friends to try it for themselves. Worth every penny.

Источников: 3
  [1] doc_id=4990, score=0.2609, name=5/5 · Авто и сервис · I keep coming back here. I really enjo…
  [2] doc_id=3406, score=0.2609, name=5/5 · Авто и сервис · I keep coming back here. I really enjo…
  [3] doc_id=2366, score=0.2609, name=5/5 · Авто и сервис · I keep coming back here. I really enjo…

--- Demo 3 (авто): «honest mechanic fast oil change» ---
Ответ:
На основании найденных отзывов:

[1] 5/5 · Авто и сервис · I keep coming back here. I really enjoyed a fast oil change…
doc_id=4898, score=0.81
Рейтинг: 5/5 звёзд. Категория: Авто и сервис.

I keep coming back here. I really enjoyed a fast oil change and an honest mechanic. Worth every penny.

[2] 5/5 · Авто и сервис · I keep coming back here. I really enjoyed a fast oil change…
doc_id=1228, score=0.81
Рейтинг: 5/5 звёзд. Категория: Авто и сервис.

I keep coming back here. I really enjoyed a fast oil change and an honest mechanic. Worth every penny.

[3] 5/5 · Авто и сервис · I keep coming back here. I really enjoyed a fast oil change…
doc_id=4003, score=0.81
Рейтинг: 5/5 звёзд. Категория: Авто и сервис.

I keep coming back here. I really enjoyed a fast oil change and an honest mechanic. Worth every penny.

Источников: 3
  [1] doc_id=4898, score=0.8065, name=5/5 · Авто и сервис · I keep coming back here. I really enjo…
  [2] doc_id=1228, score=0.8065, name=5/5 · Авто и сервис · I keep coming back here. I really enjo…
  [3] doc_id=4003, score=0.8065, name=5/5 · Авто и сервис · I keep coming back here. I really enjo…

--- Negative: «how do I configure a wifi router» ---
Ответ:
В базе отзывов не найдено релевантных фрагментов. Ответить по данным невозможно.

Источников: 3
  [1] doc_id=4999, score=0.0000, name=2/5 · Бары и ночная жизнь · Such a letdown. Unfortunately, t…
  [2] doc_id=4998, score=0.0000, name=1/5 · Бары и ночная жизнь · Worst experience I've had in a w…
  [3] doc_id=4997, score=0.0000, name=4/5 · Авто и сервис · Hands down one of my favorites. I real…
```

## eval.py
```
=== Оценка retrieval-качества (k=3, вопросов=16) ===

Бэкенд          Recall@k     MRR@k
----------------------------------
TF-IDF             0.875     0.875

[i] Собран только один бэкенд. Соберите оба (установите sentence-transformers и пересоберите индекс), чтобы увидеть сравнение.
```

## pytest
```
rootdir: /home/claude/yelp-rag
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 15 items

tests/test_chunking.py::test_chunk_text_respects_max_size PASSED         [  6%]
tests/test_chunking.py::test_chunk_text_splits_by_paragraphs PASSED      [ 13%]
tests/test_chunking.py::test_chunk_text_overlap_between_chunks PASSED    [ 20%]
tests/test_chunking.py::test_chunk_text_empty_returns_empty PASSED       [ 26%]
tests/test_chunking.py::test_chunk_document_has_doc_id PASSED            [ 33%]
tests/test_chunking.py::test_run_creates_chunks_jsonl PASSED             [ 40%]
tests/test_eval.py::test_category_matches_is_case_insensitive_substring PASSED [ 46%]
tests/test_eval.py::test_evaluate_perfect_recall_and_mrr PASSED          [ 53%]
tests/test_eval.py::test_evaluate_counts_miss PASSED                     [ 60%]
tests/test_retrieval.py::test_search_returns_k_results PASSED            [ 66%]
tests/test_retrieval.py::test_search_results_have_required_fields PASSED [ 73%]
tests/test_retrieval.py::test_search_prefers_matching_category PASSED    [ 80%]
tests/test_retrieval.py::test_search_empty_query_returns_empty PASSED    [ 86%]
tests/test_retrieval.py::test_ask_sources_contain_doc_id PASSED          [ 93%]
tests/test_retrieval.py::test_ask_refuses_without_relevant_context PASSED [100%]

============================== 15 passed in 1.39s ==============================
```
