"""
Synthetic E-Commerce Review Dataset Generator
================================================
Generates a CSV of realistic-style e-commerce product reviews with star
ratings (1-5), built from template fragments combined with randomized
product nouns, adjectives, and complaint/praise details. This produces
varied, non-repetitive review text without depending on an external
dataset download.
"""

import random
import csv

PRODUCTS = [
    "blender", "phone case", "backpack", "running shoes", "coffee maker",
    "bluetooth speaker", "office chair", "yoga mat", "desk lamp", "water bottle",
    "wireless mouse", "winter jacket", "kitchen knife set", "laptop stand",
    "electric toothbrush", "air fryer", "bookshelf", "garden hose", "sunglasses",
    "wireless earbuds", "phone charger", "throw pillow", "bath towel set",
    "skincare cream", "board game", "vacuum cleaner", "camping tent", "wallet",
]

POSITIVE_TEMPLATES = [
    "This {product} exceeded my expectations. {detail}",
    "I absolutely love this {product}. {detail}",
    "Great {product} for the price. {detail}",
    "Highly recommend this {product}! {detail}",
    "Best {product} I've purchased this year. {detail}",
    "Works perfectly and looks great. {detail}",
    "Excellent quality {product}, arrived on time. {detail}",
    "Very happy with this purchase. {detail}",
    "Five stars, this {product} is amazing. {detail}",
    "Superb build quality and fast shipping. {detail}",
]

POSITIVE_DETAILS = [
    "Fast shipping and well packaged.",
    "Customer service was also fantastic when I had a question.",
    "Easy to set up and use right out of the box.",
    "The material feels durable and well made.",
    "It does exactly what it says and more.",
    "Would definitely buy again from this seller.",
    "My whole family loves it.",
    "Better than the more expensive brand I tried before.",
    "Looks even nicer in person than in the photos.",
    "No complaints at all, works flawlessly.",
]

NEGATIVE_TEMPLATES = [
    "The {product} arrived broken and I am very unhappy.",
    "Terrible quality {product}, broke after one use. {detail}",
    "Very disappointed with this {product}. {detail}",
    "Do not buy this {product}, total waste of money. {detail}",
    "The {product} stopped working within a week. {detail}",
    "Poor packaging led to a damaged {product} on arrival. {detail}",
    "This {product} is nothing like the description. {detail}",
    "Worst purchase I've made in a long time. {detail}",
    "Cheaply made and overpriced for what you get. {detail}",
    "I regret buying this {product}. {detail}",
]

NEGATIVE_DETAILS = [
    "Customer support never responded to my emails.",
    "It stopped working after just a few days.",
    "The material feels flimsy and cheap.",
    "Nothing like the pictures shown on the listing.",
    "I'm requesting a refund immediately.",
    "Save your money and look elsewhere.",
    "Very disappointed, expected much better quality.",
    "Arrived with visible damage and missing parts.",
    "Completely unusable straight out of the box.",
    "I would not recommend this to anyone.",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay, does the job but nothing special. {detail}",
    "Average {product}, meets basic expectations. {detail}",
    "It's fine for the price, not great not terrible. {detail}",
    "Decent {product} overall, a few minor issues. {detail}",
    "Works as expected, no strong feelings either way. {detail}",
]

NEUTRAL_DETAILS = [
    "Shipping took longer than expected.",
    "Some assembly required that wasn't mentioned.",
    "It's a reasonable option if you're on a budget.",
    "Might consider other brands next time.",
    "Does what it's supposed to, nothing more.",
]


def make_review(rng, rating):
    product = rng.choice(PRODUCTS)
    if rating >= 4:
        template = rng.choice(POSITIVE_TEMPLATES)
        detail = rng.choice(POSITIVE_DETAILS)
    elif rating <= 2:
        template = rng.choice(NEGATIVE_TEMPLATES)
        detail = rng.choice(NEGATIVE_DETAILS)
    else:
        template = rng.choice(NEUTRAL_TEMPLATES)
        detail = rng.choice(NEUTRAL_DETAILS)
    text = template.format(product=product, detail=detail)
    return text


def generate_reviews_csv(path="ecommerce_reviews.csv", n_rows=1500, seed=42, missing_frac=0.02):
    rng = random.Random(seed)
    # Rating distribution skewed realistically: more 5s and 1s than 3s,
    # mirroring typical e-commerce review polarization.
    rating_weights = {1: 0.16, 2: 0.10, 3: 0.12, 4: 0.22, 5: 0.40}
    ratings_pool = list(rating_weights.keys())
    weights = list(rating_weights.values())

    rows = []
    for i in range(n_rows):
        rating = rng.choices(ratings_pool, weights=weights, k=1)[0]
        text = make_review(rng, rating)
        rows.append({"review_id": i + 1, "review_text": text, "rating": rating})

    # Inject a small fraction of missing review_text values, mirroring
    # real-world messy data (blank submissions, failed form fields, etc.)
    n_missing = int(n_rows * missing_frac)
    missing_indices = rng.sample(range(n_rows), n_missing)
    for idx in missing_indices:
        rows[idx]["review_text"] = ""

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "review_text", "rating"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n_rows} reviews ({n_missing} with missing text) -> {path}")


if __name__ == "__main__":
    generate_reviews_csv()
