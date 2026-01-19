# ============================================================
# 🚀 TRAIN MODEL — YOUTUBE SHORTS VIRALITY AI
# ============================================================

import pandas as pd
import numpy as np
import re
import joblib
from collections import defaultdict
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.sparse import hstack
import xgboost as xgb

print("📥 Chargement du dataset...")

dataset = load_dataset(
    "tarekmasryo/YouTube-Shorts-TikTok-Trends-2025",
    data_files="data/youtube_shorts_tiktok_trends_2025.csv",
    split="train"
)

df = pd.DataFrame(dataset)

# Nettoyage colonnes texte
for col in ["title", "description", "hashtags"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

# Durée
df["duration"] = pd.to_numeric(
    df.get("duration_sec", 30), errors="coerce"
).fillna(30)

# Vues
df["views"] = pd.to_numeric(df["views"], errors="coerce")
df = df[(df["views"] > 0) & (df["duration"] <= 60)].reset_index(drop=True)

# Texte combiné
df["text_all"] = df["title"] + " " + df["description"] + " " + df["hashtags"]

print(f"🎯 Shorts utilisés : {len(df)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["title_len"] = df["title"].str.len()
df["desc_len"] = df["description"].str.len()
df["hashtag_count"] = df["hashtags"].apply(lambda x: len(x.split()))

# ============================================================
# VECTORIZATION
# ============================================================

tfidf = TfidfVectorizer(
    max_features=7000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_text = tfidf.fit_transform(df["text_all"])

scaler = MinMaxScaler()
X_num = scaler.fit_transform(
    df[["duration", "title_len", "desc_len", "hashtag_count"]]
)

X = hstack([X_text, X_num])
y = np.log1p(df["views"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# MODEL
# ============================================================

model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1
)

print("⏳ Entraînement du modèle...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n📊 ÉVALUATION")
print("MAE :", int(mean_absolute_error(np.expm1(y_test), np.expm1(pred))))
print("R²  :", round(r2_score(y_test, pred), 3))

# ============================================================
# 🔥 KEYWORDS STATISTICS (DATASET-DRIVEN)
# ============================================================

print("\n📊 Calcul des statistiques keywords...")

word_views = defaultdict(list)

for _, row in df.iterrows():
    text = f"{row['title']} {row['description']} {row['hashtags']}".lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = set(text.split())
    for w in words:
        if len(w) > 3:
            word_views[w].append(row["views"])

keyword_stats = {
    w: int(np.mean(v))
    for w, v in word_views.items()
    if len(v) >= 5
}

# ============================================================
# SAVE
# ============================================================

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(keyword_stats, "keyword_stats.pkl")

print("✅ model.pkl / tfidf.pkl / scaler.pkl / keyword_stats.pkl sauvegardés")
