import pandas as pd
import re
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
client = MongoClient(DATABASE_URL)
db = client["store"]

stemmer = StemmerFactory().create_stemmer()
stopwords_factory = StopWordRemoverFactory()
stopwords = set(stopwords_factory.get_stop_words())

if 'untuk' in stopwords:
    stopwords.remove('untuk')

def clean_text(text):
    text = re.sub(r'[^\w\s\-]', '', str(text).lower())
    return text
def tokenize(text):
    return text.split()
def remove_stopwords(words):
    return [w for w in words if w not in stopwords]
def stem_words(words):
    return [stemmer.stem(w) for w in words]

def preprocess_text(text):
    text = clean_text(text)
    words = tokenize(text)
    words = remove_stopwords(words)
    words = stem_words(words)
    return ' '.join(words)

def normalize_column(series):
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())

def calculate_column_weights_tfidf(df, columns, age_boost=3.0):
    weights = {}
    for col in columns:
        processed = df[col].fillna('').astype(str).apply(preprocess_text)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(processed)
        importance = tfidf_matrix.sum(axis=1).mean()
        weights[col] = float(importance)
    if 'age_name' in weights:
        weights['age_name'] *= age_boost
    return weights

def compute_variation_weight(series):
    if series.max() == 0:
        return 0.0
    return float(series.std() / series.max())

def weighted_text(text, weight, multiplier=10):
    processed = preprocess_text(text)
    return ' '.join([processed] * max(1, int(weight * multiplier)))

def preprocess_data():
    product_cursor = db["Product"].find(
        {"isArchived": False},  # hanya ambil yang tidak diarsipkan
        {
            "name": 1,
            "description": 1,
            "ageId": 1,
            "categories": 1,
            "price": 1,
            "stock": 1
        }
    )
    age_cursor = db["Age"].find()

    product_df = pd.DataFrame(list(product_cursor))
    age_df = pd.DataFrame(list(age_cursor))

    product_df = product_df.merge(age_df, left_on='ageId', right_on='_id', how='left')
    product_df.rename(columns={'_id_x': 'id', 'name_x': 'name', 'name_y': 'age_name'}, inplace=True)
    product_df.drop(columns=['_id_y'], inplace=True)

    product_df['price_norm'] = normalize_column(product_df['price'])
    product_df['stock_norm'] = normalize_column(product_df['stock'])

    text_columns = ['name', 'description', 'categories', 'age_name']
    text_weights = calculate_column_weights_tfidf(product_df, text_columns, age_boost=3.0)
    numeric_weights = {
        'price': compute_variation_weight(product_df['price']),
        'stock': compute_variation_weight(product_df['stock'])
    }

    total_text_weight = sum(text_weights.values())
    total_numeric_weight = sum(numeric_weights.values())

    text_scale = 0.5 / total_text_weight
    numeric_scale = 0.5 / total_numeric_weight

    for col in text_weights:
        text_weights[col] *= text_scale
    for col in numeric_weights:
        numeric_weights[col] *= numeric_scale

    def combine_features(row):
        combined = ''
        for col in text_columns:
            content = ' '.join(row[col]) if isinstance(row[col], list) else str(row[col])
            combined += weighted_text(content, text_weights.get(col, 1.0), multiplier=20) + ' '

        price_text = f"harga_{int(row['price_norm'] * 10)}"
        stock_text = f"stok_{int(row['stock_norm'] * 10)}"

        combined += ' '.join([price_text] * max(1, int(numeric_weights['price'] * 20))) + ' '
        combined += ' '.join([stock_text] * max(1, int(numeric_weights['stock'] * 20)))
        return combined.strip()

    product_df['tags'] = product_df.apply(combine_features, axis=1)

    tfidf = TfidfVectorizer()
    vector = tfidf.fit_transform(product_df['tags'])
    similarity = cosine_similarity(vector)

    return product_df, similarity

product_df, similarity = preprocess_data()

image_df = pd.DataFrame(list(db["Image"].find({}, {"productId": 1, "url": 1})))

def get_images(product_id):
    matched = image_df[image_df['productId'] == ObjectId(product_id)]
    return matched['url'].tolist()

def get_recommendations_by_keyword(keyword):
    keywords = keyword.lower().split()
    stemmed_keywords = [stemmer.stem(k) for k in keywords]

    matched_indices = product_df[
        product_df['tags'].apply(lambda x: all(kw in x for kw in stemmed_keywords))
    ].index.tolist()

    if not matched_indices:
        return {"matched_products": [], "recommended_products": []}

    matched_age_names = product_df.loc[matched_indices, 'age_name'].unique()

    avg_similarity = sum(similarity[i] for i in matched_indices) / len(matched_indices)

    prioritized_indices = product_df[
        product_df['age_name'].isin(matched_age_names)
    ].index.tolist()

    distances = sorted(
        [(idx, sim) for idx, sim in enumerate(avg_similarity) if idx not in matched_indices and idx in prioritized_indices],
        key=lambda x: x[1],
        reverse=True
    )

    recommended_indices = [idx for idx, _ in distances][:20]

    return format_recommendation_results(matched_indices, recommended_indices)

def get_recommendations_by_purchased_products():
    completed_orders = list(db["Order"].find({"status": "COMPLETED"}, {"_id": 1}))
    completed_order_ids = [order["_id"] for order in completed_orders]

    if not completed_order_ids:
        return {"matched_products": [], "recommended_products": []}

    order_items = db["OrderItem"].find({"orderId": {"$in": completed_order_ids}}, {"productId": 1})
    product_ids = {item["productId"] for item in order_items if "productId" in item}

    if not product_ids:
        return {"matched_products": [], "recommended_products": []}

    purchased_df = product_df[product_df['id'].isin(product_ids)]
    if purchased_df.empty:
        return {"matched_products": [], "recommended_products": []}

    purchased_indices = purchased_df.index.tolist()
    purchased_age_names = purchased_df['age_name'].unique()

    avg_similarity = sum(similarity[i] for i in purchased_indices) / len(purchased_indices)

    prioritized_indices = product_df[
        product_df['age_name'].isin(purchased_age_names)
    ].index.tolist()

    distances = sorted(
        [(idx, sim) for idx, sim in enumerate(avg_similarity) if idx not in purchased_indices and idx in prioritized_indices],
        key=lambda x: x[1],
        reverse=True
    )

    recommended_indices = [idx for idx, _ in distances][:20]

    return format_recommendation_results(purchased_indices, recommended_indices)

def format_recommendation_results(matched_indices, recommended_indices):
    matched_df = product_df.iloc[matched_indices].copy()
    recommended_df = product_df.iloc[recommended_indices].copy()

    for df in [matched_df, recommended_df]:
        df['id'] = df['id'].astype(str)
        df['images'] = df['id'].apply(get_images)

    return {
        "matched_products": matched_df[[
            'id', 'name', 'description', 'categories',
            'age_name', 'price', 'images', 'stock'
        ]].to_dict(orient="records"),
        "recommended_products": recommended_df[[
            'id', 'name', 'description', 'categories',
            'age_name', 'price', 'images', 'stock'
        ]].to_dict(orient="records")
    }
    
result = get_recommendations_by_keyword('sabun buat orang dewasa')