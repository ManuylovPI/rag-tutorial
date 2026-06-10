import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever import Retriever


def print_hit(i: int, hit: dict) -> None:
    preview = hit["text"][:120].replace("\n", " ")
    print(f"  [{i}] doc_id={hit['doc_id']}, score={hit['score']:.4f}")
    print(f"      {preview}…")


def main() -> None:
    print("=== Проверка Retriever (итерация 5) ===\n")

    r = Retriever()
    print(f"OK: индекс загружен, активный бэкенд = {r.backend_name}")
    print(f"Чанков в индексе: {len(r.chunks)}\n")

    queries = [
        ("rude bartender ignored us at the bar",
         "слова есть в данных -> score > 0, категория «Бары»"),
        ("delicious pizza and friendly waiter",
         "слова есть в данных -> score > 0, категория «Рестораны»"),
        ("how to configure a wifi router at home",
         "слов нет в отзывах -> низкий score / отказ на стороне генератора"),
    ]

    for query, hint in queries:
        print(f"Запрос: «{query}»")
        print(f"Ожидание: {hint}")
        results = r.search(query, k=3)
        print(f"Получено результатов: {len(results)}")
        for i, hit in enumerate(results, 1):
            print_hit(i, hit)
        print()

    print("=== Итог ===")
    print("Если для каждого запроса видно 3 результата с полями "
          "doc_id / score / text — итерация 5 работает.")


if __name__ == "__main__":
    main()
