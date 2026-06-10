import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    EMBEDDINGS_NPY,
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    TOP_K,
    VECTORIZER_PKL,
)
from app.generator import ask
from app.prompts import min_score_for
from app.retriever import Retriever

DEMO_QUESTIONS = [
    "rude bartender and watered down drinks",
    "delicious food and great service",
    "honest mechanic fast oil change",
    "how do I configure a wifi router", 
]


def index_exists() -> bool:
    chunks_ok = INDEX_CHUNKS_JSONL.exists()
    tfidf_ok = VECTORIZER_PKL.exists() and MATRIX_NPZ.exists()
    semantic_ok = EMBEDDINGS_NPY.exists()
    return chunks_ok and (tfidf_ok or semantic_ok)


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever()

@st.cache_resource
def load_retriever() -> Retriever:
    r = Retriever()
    try:
        r.search("warmup", k=1)
    except Exception:
        pass
    return r


def render_chunk(i: int, src: dict, expanded: bool = True) -> None:
    label = f"[{i}] doc_id={src['doc_id']} · score={src['score']:.4f}"
    with st.expander(label, expanded=expanded):
        st.markdown(f"**{src['name']}**")
        st.text(src["text"])


def render_fragments(sources: list[dict], min_score: float) -> None:
    st.subheader("Найденные фрагменты (top-k)")
    if not sources:
        st.info("Фрагменты не найдены.")
        return
    for i, src in enumerate(sources, 1):
        render_chunk(i, src, expanded=src["score"] >= min_score)


def render_sources(sources: list[dict]) -> None:
    st.subheader("Источники")
    if not sources:
        st.info("Источники отсутствуют.")
        return
    for i, src in enumerate(sources, 1):
        render_chunk(i, src, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Yelp RAG", layout="wide")
    st.title("Yelp Reviews RAG")
    st.caption("Учебный RAG по отзывам Yelp: поиск + demo-ответ с источниками")

    if not index_exists():
        st.error(
            "Индекс не собран. Сначала выполните:\n\n"
            "`uv run python scripts/prepare_datasets.py`\n\n"
            "`uv run python scripts/build_index.py`"
        )
        st.stop()

    retriever = load_retriever()
    min_score = min_score_for(retriever.backend_name)

    st.sidebar.header("Статус")
    badge = "🧠 semantic" if retriever.backend_name == "semantic" else "🔤 TF-IDF"
    st.sidebar.success(f"Активный бэкенд: {badge}")
    st.sidebar.caption(f"Чанков в индексе: {len(retriever.chunks)}")

    st.sidebar.header("Настройки")
    k = st.sidebar.slider("top-k", min_value=1, max_value=10, value=TOP_K)
    user_min_score = st.sidebar.slider(
        "Порог релевантности (min score)",
        min_value=0.0,
        max_value=1.0,
        value=float(min_score),
        step=0.01,
        help="Фрагменты ниже порога не попадают в ответ (срабатывает отказ).",
    )

    st.sidebar.header("Demo-вопросы")
    for q in DEMO_QUESTIONS:
        if st.sidebar.button(q, use_container_width=True):
            st.session_state["question"] = q

    question = st.text_input("Ваш вопрос", key="question")

    if st.button("Спросить", type="primary"):
        if not question.strip():
            st.warning("Введите вопрос.")
            st.stop()

        with st.spinner(f"Поиск ({retriever.backend_name})…"):
            result = ask(question.strip(), k=k, retriever=retriever)

        render_fragments(result["sources"], user_min_score)

        st.subheader("Ответ")
        relevant = [s for s in result["sources"] if s["score"] >= user_min_score]
        if not relevant:
            st.warning(
                "В базе отзывов не найдено достаточно релевантных фрагментов "
                "(все ниже порога). Ответить по данным невозможно."
            )
        else:
            st.text(result["answer"])

        render_sources(result["sources"])


if __name__ == "__main__":
    main()
