# Рабочий процесс (workflow)

## Цикл разработки

1. Взять одну итерацию из tasklist.
2. Реализовать минимальный код для проверяемого результата.
3. Запустить проверку итерации (скрипт или pytest).
4. Только после зелёной проверки переходить к следующей итерации.

## Воспроизведение проекта с нуля

```bash
uv venv && uv sync
uv run python scripts/prepare_datasets.py --limit 5000   # данные
uv run python scripts/build_index.py                     # индекс
uv run pytest tests/ -v                                  # тесты
uv run python scripts/check_retrieval.py                 # поиск
uv run python scripts/check_generator.py                 # demo + negative
uv run python scripts/eval.py                            # метрики
uv run streamlit run app/main.py                         # UI
```

## Переключение бэкенда

```bash
RAG_BACKEND=tfidf    uv run python scripts/check_generator.py
RAG_BACKEND=semantic uv run python scripts/check_generator.py
```

## Критерий готовности

Любой шаг воспроизводится по командам выше на чистой машине; результаты совпадают
с логами в docs/logs/.
