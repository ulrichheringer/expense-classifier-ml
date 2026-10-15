"""Expense-description categorization with TF-IDF and Multinomial Naive Bayes."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = 42
DESCRIPTIONS = {
    "food": ["grocery market weekly shop", "restaurant dinner receipt", "coffee shop purchase", "bakery breakfast order", "supermarket produce and pantry", "lunch cafe payment", "food delivery order", "corner store groceries"],
    "transport": ["metro transit card recharge", "taxi ride to office", "fuel station petrol purchase", "parking garage fee", "bus ticket commute", "rideshare trip receipt", "train fare reservation", "toll road payment"],
    "utilities": ["monthly electricity bill", "water service invoice", "home internet broadband bill", "mobile phone plan charge", "gas utility statement", "electric provider payment", "fiber connection subscription", "waste collection service"],
    "housing": ["monthly apartment rent", "landlord rent payment", "home repair maintenance visit", "building association fee", "property management charge", "lease deposit payment", "plumber apartment repair", "home insurance premium"],
    "health": ["pharmacy prescription medicine", "clinic doctor appointment", "dental care visit", "health laboratory blood test", "medical insurance copay", "optician eye exam", "hospital consultation fee", "prescription refill receipt"],
    "subscriptions": ["video streaming monthly plan", "cloud software license renewal", "music service membership charge", "online storage subscription", "developer tool monthly subscription", "digital news membership renewal", "design software annual plan", "fitness app premium access"],
}


def _dataset(seed=SEED):
    texts, labels, groups = [], [], []
    for category, templates in DESCRIPTIONS.items():
        for template_id, text in enumerate(templates):
            for receipt in range(10):
                texts.append(f"{text}; receipt {receipt}; card purchase")
                labels.append(category)
                groups.append(f"{category}-{template_id}")
    return texts, labels, groups


def _model(seed=SEED):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), MultinomialNB(alpha=0.25))


def sample_input():
    return {"description": "supermarket grocery shop and produce"}


def train_model(seed=SEED):
    texts, labels, _ = _dataset(seed)
    return _model(seed).fit(texts, labels)


def predict(model, payload):
    text = payload["description"] if isinstance(payload, dict) else str(payload)
    return {"category": str(model.predict([text])[0]),
            "probabilities": {str(name): round(float(value), 4) for name, value in zip(model.classes_, model.predict_proba([text])[0])}}


def evaluate(seed=SEED):
    texts, labels, groups = _dataset(seed)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(texts, labels, groups))
    model = _model(seed).fit([texts[i] for i in tr], [labels[i] for i in tr])
    pred = model.predict([texts[i] for i in te])
    return {"accuracy": round(float(accuracy_score([labels[i] for i in te], pred)), 4),
            "macro_f1": round(float(f1_score([labels[i] for i in te], pred, average="macro")), 4),
            "test_rows": int(len(te)), "categories": len(DESCRIPTIONS)}
