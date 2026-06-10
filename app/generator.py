from app.config import TOP_K
from app.prompts import REFUSAL_EMPTY_QUESTION, REFUSAL_NO_CONTEXT, min_score_for
from app.retriever import Retriever


def build_answer(hits: list[dict], min_score: float) -> str:
    relevant = [h for h in hits if h["score"] >= min_score]
    if not relevant:
        return REFUSAL_NO_CONTEXT

    parts = ["На основании найденных отзывов:"]
    for i, hit in enumerate(relevant, 1):
        parts.append(f"\n[{i}] {hit['name']}")
        parts.append(f"doc_id={hit['doc_id']}, score={hit['score']:.2f}")
        parts.append(hit["text"])
    return "\n".join(parts)


def format_sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": hit["doc_id"],
            "name": hit.get("name", ""),
            "text": hit["text"],
            "score": hit["score"],
        }
        for hit in hits
    ]


def ask(
    question: str,
    k: int = TOP_K,
    retriever: Retriever | None = None,
) -> dict:
    if not question.strip():
        return {"answer": REFUSAL_EMPTY_QUESTION, "sources": []}

    r = retriever or Retriever()
    hits = r.search(question.strip(), k=k)
    min_score = min_score_for(r.backend_name)
    return {
        "answer": build_answer(hits, min_score),
        "sources": format_sources(hits),
    }
