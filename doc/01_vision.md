# Видение (vision)

## Архитектура

```
datasets.json → documents.jsonl → chunks.jsonl → index/ → Retriever → Generator → UI
```

- **Ingest** (`scripts/ingest.py`): нормализация и метаданные.
- **Chunking** (`app/chunker.py`): нарезка по абзацам с overlap.
- **Index** (`scripts/build_index.py`): TF-IDF и semantic эмбеддинги.
- **Retriever** (`app/retriever.py`): единый интерфейс над двумя бэкендами,
  поиск top-k по cosine similarity.
- **Generator** (`app/generator.py`): ответ из чанков + отказ по порогу.
- **UI** (`app/main.py`): Streamlit, индикатор бэкенда, порог, top-k.

## Ключевое архитектурное решение

Retriever поддерживает два взаимозаменяемых бэкенда (TF-IDF и semantic) за единым
интерфейсом. Это позволяет: (1) сохранить совместимость с образцом, (2) честно
сравнить лексический и семантический поиск на одних данных, (3) безопасно
откатываться на TF-IDF, если модель эмбеддингов недоступна.

## Критерий готовности

Любой компонент можно проверить отдельным скриптом (check_retrieval,
check_generator, eval) и тестами. Замена бэкенда — одной переменной окружения.
