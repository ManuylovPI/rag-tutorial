import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generator import ask


def show(label: str, question: str) -> None:
    print(f"\n--- {label}: «{question}» ---")
    result = ask(question)
    print(f"Ответ:\n{result['answer']}\n")
    print(f"Источников: {len(result['sources'])}")
    for i, src in enumerate(result["sources"], 1):
        print(
            f"  [{i}] doc_id={src['doc_id']}, score={src['score']:.4f}, "
            f"name={src['name'][:60]}…"
        )


if __name__ == "__main__":
    show("Demo 1 (бары)", "rude bartender and watered down drinks")
    show("Demo 2 (рестораны)", "delicious food and great service")
    show("Demo 3 (авто)", "honest mechanic fast oil change")
    show("Negative", "how do I configure a wifi router")
