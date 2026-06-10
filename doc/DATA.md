# Описание данных (DATA.md)

## Источник

- **Датасет:** [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full)
- **Происхождение:** Yelp Dataset Challenge 2015 (Xiang Zhang, Junbo Zhao,
  Yann LeCun. *Character-level Convolutional Networks for Text Classification*,
  NIPS 2015).
- **Объём оригинала:** 650 000 обучающих + 50 000 тестовых отзывов.
- **Язык:** преимущественно английский.
- **Лицензия:** Yelp Dataset License (учебное/исследовательское использование).

## Что берём

Случайный срез из **5000 отзывов** (фиксированный `seed=42` для
воспроизводимости). Скрипт: `scripts/prepare_datasets.py`.

Масштаб выбран так, чтобы заведомо превысить порог оценки «Отлично»: 5000
записей дают **6316 чанков** (требуется 1000+ записей или чанков).

## Структура записи `datasets.json`

```json
{
  "id": 0,
  "name": "5/5 · Рестораны и еда · What a pleasant surprise...",
  "text": "Рейтинг: 5/5 звёзд. Категория: Рестораны и еда.\n\n<полный текст отзыва>"
}
```

- `id` — порядковый номер (становится `doc_id`).
- `name` — человекочитаемый заголовок: рейтинг + категория + превью текста.
  Используется в UI и источниках.
- `text` — индексируемое содержимое: строка с рейтингом и категорией + сам отзыв.

## Что именно индексируется

Индексируется поле `text` целиком. Оно включает:

1. **Рейтинг** (1–5 звёзд) — из метки датасета (`label` 0–4 → звёзды 1–5).
2. **Категория заведения** — выводится эвристически по ключевым словам в тексте
   (рестораны, кафе, бары, салоны, авто, отели, магазины, медицина, прочее),
   т.к. в Yelp Review Full отдельного поля категории нет.
3. **Текст отзыва** — основной контент для поиска.

## Преобразования (pipeline)

1. `prepare_datasets.py` → `data/raw/datasets.json` (5000 записей).
2. `ingest.py` → `data/processed/documents.jsonl` (нормализация пробелов,
   добавление `doc_id`, `source_file`).
3. `chunker.py` → `data/processed/chunks.jsonl` (нарезка по абзацам,
   `max_chars=400`, `overlap=50`).
4. `build_index.py` → `data/index/` (TF-IDF: `vectorizer.pkl` + `matrix.npz`;
   semantic: `embeddings.npy` + `embed_model.txt`; общий снимок `chunks.jsonl`).

## Воспроизводимость

```bash
# Реальные данные (нужен доступ к Hugging Face):
uv run python scripts/prepare_datasets.py --limit 5000 --seed 42

# Офлайн-фолбэк (синтетический корпус того же формата):
uv run python scripts/prepare_datasets.py --synthetic --limit 5000 --seed 42
```
