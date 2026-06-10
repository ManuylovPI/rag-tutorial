import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EMBEDDINGS_NPY, MATRIX_NPZ
from app.retriever import Retriever

GOLD = [
    ("the bartender was rude and ignored us", "Бары"),
    ("watered down cocktails and sticky tables", "Бары"),
    ("delicious pasta and a great waiter", "Рестораны"),
    ("the pizza was cold and the order was wrong", "Рестораны"),
    ("smooth latte and a flaky croissant", "Кафе"),
    ("the espresso tasted burnt", "Кафе"),
    ("fast oil change and an honest mechanic", "Авто"),
    ("they scratched my car during the repair", "Авто"),
    ("spotless hotel room and helpful front desk", "Отели"),
    ("the room smelled of smoke", "Отели"),
    ("a fantastic haircut and a relaxing massage", "Салоны"),
    ("the cashier was rude and refused my refund", "Магазины"),
    ("the person mixing drinks completely neglected the customers", "Бары"),
    ("the meal was tasty and the server was attentive", "Рестораны"),
    ("quick car servicing by a trustworthy specialist", "Авто"),
    ("a tidy place to sleep with courteous reception staff", "Отели"),
]


def category_matches(name: str, expected: str) -> bool:
    return expected.lower() in name.lower()


def evaluate(retriever: Retriever, k: int) -> tuple[float, float]:
    hits_at_k = 0
    reciprocal_ranks = 0.0

    for question, expected_cat in GOLD:
        results = retriever.search(question, k=k)
        rank = None
        for i, hit in enumerate(results, 1):
            if category_matches(hit["name"], expected_cat):
                rank = i
                break
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks += 1.0 / rank

    n = len(GOLD)
    return hits_at_k / n, reciprocal_ranks / n


def run_backend(backend: str, k: int) -> tuple[float, float] | None:
    try:
        retriever = Retriever(backend=backend)
    except Exception as exc:  # noqa: BLE001
        print(f"[i] Бэкенд «{backend}» недоступен: {exc}")
        return None
    if retriever.backend_name != backend:
        return None
    return evaluate(retriever, k)


def main() -> None:
    k = 3
    print(f"=== Оценка retrieval-качества (k={k}, вопросов={len(GOLD)}) ===\n")

    rows = []
    if MATRIX_NPZ.exists():
        res = run_backend("tfidf", k)
        if res:
            rows.append(("TF-IDF", *res))
    if EMBEDDINGS_NPY.exists():
        res = run_backend("semantic", k)
        if res:
            rows.append(("semantic", *res))

    if not rows:
        print("Нет собранных индексов. Сначала: uv run python scripts/build_index.py")
        return

    print(f"{'Бэкенд':<12}{'Recall@k':>12}{'MRR@k':>10}")
    print("-" * 34)
    for name, recall, mrr in rows:
        print(f"{name:<12}{recall:>12.3f}{mrr:>10.3f}")

    if len(rows) == 2:
        d_recall = rows[1][1] - rows[0][1]
        d_mrr = rows[1][2] - rows[0][2]
        print(
            f"\nΔ semantic − TF-IDF:  Recall@k {d_recall:+.3f},  MRR@k {d_mrr:+.3f}"
        )
        print(
            "Положительная дельта подтверждает, что semantic-поиск лучше "
            "находит релевантные отзывы, особенно на перефразированных вопросах."
        )
    else:
        print(
            "\n[i] Собран только один бэкенд. Соберите оба "
            "(установите sentence-transformers и пересоберите индекс), "
            "чтобы увидеть сравнение."
        )


if __name__ == "__main__":
    main()
