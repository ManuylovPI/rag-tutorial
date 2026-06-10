import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RAW_DATASETS

LABEL_TO_STARS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

CATEGORY_KEYWORDS = {
    "Рестораны и еда": [
        "food", "restaurant", "menu", "dish", "pizza", "burger", "sushi",
        "breakfast", "lunch", "dinner", "tasty", "delicious", "flavor",
        "chef", "waiter", "waitress", "appetizer", "dessert", "meal",
    ],
    "Кафе и напитки": [
        "coffee", "latte", "espresso", "cafe", "barista", "tea", "bakery",
        "pastry", "croissant", "brunch", "smoothie",
    ],
    "Бары и ночная жизнь": [
        "bar", "beer", "cocktail", "drinks", "pub", "wine", "happy hour",
        "bartender", "nightlife", "lounge",
    ],
    "Салоны и услуги красоты": [
        "salon", "haircut", "nails", "manicure", "spa", "massage",
        "stylist", "barber", "facial",
    ],
    "Авто и сервис": [
        "car", "tire", "mechanic", "oil change", "repair", "auto",
        "dealership", "vehicle", "brakes",
    ],
    "Отели и путешествия": [
        "hotel", "room", "stay", "resort", "check-in", "lobby", "suite",
        "motel", "front desk", "booking",
    ],
    "Магазины и шопинг": [
        "store", "shop", "purchase", "price", "shopping", "retail",
        "boutique", "mall", "cashier", "refund",
    ],
    "Здоровье и медицина": [
        "doctor", "dentist", "clinic", "appointment", "nurse", "medical",
        "hospital", "pharmacy", "patient",
    ],
}


def guess_category(text: str) -> str:
    low = text.lower()
    best_cat = "Прочие услуги"
    best_hits = 0
    for cat, words in CATEGORY_KEYWORDS.items():
        hits = sum(1 for w in words if w in low)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
    return best_cat


def make_title(stars: int, category: str, text: str) -> str:
    preview = " ".join(text.split())[:60].strip()
    return f"{stars}/5 · {category} · {preview}…"


def to_record(idx: int, stars: int, text: str) -> dict:
    text = " ".join(text.split()).strip()
    category = guess_category(text)
    name = make_title(stars, category, text)
    body = f"Рейтинг: {stars}/5 звёзд. Категория: {category}.\n\n{text}"
    return {"id": idx, "name": name, "text": body}


def build_from_huggingface(limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    print(f"Загрузка Yelp/yelp_review_full с Hugging Face (limit={limit})…")
    ds = load_dataset("Yelp/yelp_review_full", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))

    records = []
    for i, row in enumerate(ds):
        stars = LABEL_TO_STARS.get(int(row["label"]), int(row["label"]) + 1)
        records.append(to_record(i, stars, row["text"]))
    return records


_VENUES = {
    "Рестораны и еда": [
        ("the pasta and the wood-fired pizza", "the kitchen was painfully slow"),
        ("a perfectly cooked steak and attentive waiter", "the steak arrived cold and overcooked"),
        ("fresh sushi and a generous portion size", "the fish tasted off and the rice was dry"),
        ("the burger was juicy and the fries crisp", "they got my order wrong twice"),
        ("an outstanding tasting menu with real flavor", "tiny portions for an outrageous price"),
    ],
    "Кафе и напитки": [
        ("a smooth latte and a flaky croissant", "the espresso was burnt and bitter"),
        ("friendly baristas and great brunch", "they ran out of oat milk and the pastry was stale"),
        ("the cold brew here is the best in town", "I waited twenty minutes for a simple drip coffee"),
    ],
    "Бары и ночная жизнь": [
        ("creative cocktails and a fun happy hour", "the bartender ignored us for half an hour"),
        ("a great craft beer selection", "watered-down drinks and sticky tables"),
        ("a cozy lounge with good wine", "the music was so loud we couldn't talk"),
    ],
    "Салоны и услуги красоты": [
        ("a fantastic haircut and a relaxing massage", "my stylist rushed and botched the color"),
        ("the manicure lasted for weeks", "the salon was dirty and the facial irritated my skin"),
    ],
    "Авто и сервис": [
        ("a fast oil change and an honest mechanic", "they quoted me double and scratched my car"),
        ("they fixed my brakes the same day", "the repair failed within a week"),
    ],
    "Отели и путешествия": [
        ("a spotless room and a helpful front desk", "the room smelled of smoke and check-in took ages"),
        ("a beautiful suite with a great view", "the booking was lost and the lobby was chaotic"),
    ],
    "Магазины и шопинг": [
        ("helpful staff and fair prices", "the cashier was rude and refused my refund"),
        ("a well-stocked boutique with great deals", "everything was overpriced and picked over"),
    ],
    "Здоровье и медицина": [
        ("a caring dentist and a short wait", "the clinic lost my appointment and the nurse was dismissive"),
        ("the doctor took time to explain everything", "I waited two hours past my appointment time"),
    ],
}

_POSITIVE_OPENERS = [
    "Absolutely loved this place.", "What a pleasant surprise.",
    "I keep coming back here.", "Hands down one of my favorites.",
    "Exceeded my expectations.",
]
_NEGATIVE_OPENERS = [
    "Deeply disappointing experience.", "I won't be returning.",
    "Save your money and go elsewhere.", "Such a letdown.",
    "Worst experience I've had in a while.",
]
_NEUTRAL_OPENERS = [
    "It was fine, nothing special.", "Mixed feelings about this one.",
    "Decent but not memorable.", "An okay visit overall.",
]
_POSITIVE_CLOSERS = [
    "Highly recommend it to anyone in the area.",
    "I'll definitely be back soon.",
    "Five stars without hesitation.",
    "Worth every penny.",
]
_NEGATIVE_CLOSERS = [
    "Management should be ashamed.",
    "Do yourself a favor and avoid it.",
    "I've asked for a refund.",
    "Never again.",
]
_NEUTRAL_CLOSERS = [
    "Might give it another chance someday.",
    "Your mileage may vary.",
    "Take that for what it's worth.",
]


def build_synthetic(limit: int, seed: int) -> list[dict]:
    print(f"Генерация синтетического корпуса отзывов (limit={limit})…")
    rng = random.Random(seed)
    categories = list(_VENUES.keys())
    records = []

    for i in range(limit):
        category = rng.choice(categories)
        positive_phrase, negative_phrase = rng.choice(_VENUES[category])
        stars = rng.choices([1, 2, 3, 4, 5], weights=[2, 2, 2, 3, 4])[0]

        if stars >= 4:
            opener = rng.choice(_POSITIVE_OPENERS)
            closer = rng.choice(_POSITIVE_CLOSERS)
            middle = f"I really enjoyed {positive_phrase}."
        elif stars <= 2:
            opener = rng.choice(_NEGATIVE_OPENERS)
            closer = rng.choice(_NEGATIVE_CLOSERS)
            middle = f"Unfortunately, {negative_phrase}."
        else:
            opener = rng.choice(_NEUTRAL_OPENERS)
            closer = rng.choice(_NEUTRAL_CLOSERS)
            middle = (
                f"On one hand I enjoyed {positive_phrase}, "
                f"but on the other hand {negative_phrase}."
            )

        filler = ""
        if rng.random() < 0.3:
            filler = (
                " The service overall set the tone for the whole visit, "
                "and the atmosphere played a big part in how I felt by the end. "
                "I came in with certain expectations and left thinking carefully "
                "about whether I would tell my friends to try it for themselves."
            )

        text = f"{opener} {middle}{filler} {closer}"
        records.append(to_record(i, stars, text))

    return records


def write_datasets(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"datasets": records}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка datasets.json из отзывов Yelp")
    parser.add_argument("--limit", type=int, default=5000, help="Сколько отзывов взять")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Сгенерировать синтетический корпус вместо загрузки с Hugging Face",
    )
    args = parser.parse_args()

    if args.synthetic:
        records = build_synthetic(args.limit, args.seed)
    else:
        try:
            records = build_from_huggingface(args.limit, args.seed)
        except Exception as exc:
            print(f"[!] Не удалось загрузить с Hugging Face: {exc}")
            print("[!] Переключаюсь на синтетический корпус (--synthetic).")
            records = build_synthetic(args.limit, args.seed)

    write_datasets(records, RAW_DATASETS)
    print(f"Записано {len(records)} записей -> {RAW_DATASETS}")


if __name__ == "__main__":
    main()
