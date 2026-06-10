SYSTEM_RULES = """
Ты отвечаешь только на основе найденных фрагментов из базы отзывов.
Не выдумывай факты, заведения, оценки и детали, которых нет в контексте.
Если релевантных фрагментов нет — явно откажись ответить.
""".strip()

REFUSAL_NO_CONTEXT = (
    "В базе отзывов не найдено релевантных фрагментов. "
    "Ответить по данным невозможно."
)

REFUSAL_EMPTY_QUESTION = "Задайте вопрос."

MIN_SCORE_TFIDF = 0.15
MIN_SCORE_SEMANTIC = 0.30


def min_score_for(backend_name: str) -> float:
    """Порог релевантности под конкретный бэкенд."""
    return MIN_SCORE_SEMANTIC if backend_name == "semantic" else MIN_SCORE_TFIDF


MIN_SCORE = MIN_SCORE_TFIDF
