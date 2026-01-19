# ============================================================
# 🚀 YOUTUBE SHORTS VIRALITY AI — FRONT-END PRO + GRAPHS
# ============================================================

import streamlit as st
import numpy as np
import re
import os
import joblib
from scipy.sparse import hstack
from collections import Counter
import plotly.express as px

# ---------------------------
# MODEL FILES
# ---------------------------
MODEL_FILE = "model.pkl"
TFIDF_FILE = "tfidf.pkl"
SCALER_FILE = "scaler.pkl"
KEYWORD_STATS_FILE = "keyword_stats.pkl"

# ---------------------------
# LOAD OR TRAIN MODEL
# ---------------------------
if all(os.path.exists(f) for f in [MODEL_FILE, TFIDF_FILE, SCALER_FILE, KEYWORD_STATS_FILE]):
    model = joblib.load(MODEL_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    scaler = joblib.load(SCALER_FILE)
    keyword_stats = joblib.load(KEYWORD_STATS_FILE)
else:
    st.warning("⚠️ Training model… this may take a few seconds")
    from train_model import train_model  # Assure-toi que train_model() retourne model, tfidf, scaler, keyword_stats
    model, tfidf, scaler, keyword_stats = train_model()
    joblib.dump(model, MODEL_FILE)
    joblib.dump(tfidf, TFIDF_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(keyword_stats, KEYWORD_STATS_FILE)
    st.success("✅ Model trained!")

# ---------------------------
# SPACY SETUP
# ---------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except:
    SPACY_OK = False

STOPWORDS = set([
    "the","a","an","and","or","with","this","that","from","your","you",
    "for","to","of","on","in","is","are","was","were","be","been"
])

# ---------------------------
# NLP UTILITIES
# ---------------------------
def clean_words(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 3]

def extract_key_phrases(text, max_phrases=6):
    phrases = []
    if SPACY_OK:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:
                phrases.append(chunk.text.lower())
    words = clean_words(text)
    bigrams = zip(words, words[1:])
    for a, b in bigrams:
        phrases.append(f"{a} {b}")
    return list(dict.fromkeys(phrases))[:max_phrases]

def score_phrases_with_dataset(phrases):
    scored = []
    for phrase in phrases:
        words = phrase.split()
        scores = [keyword_stats.get(w, 0) for w in words]
        avg_score = int(sum(scores) / max(1, len(scores)))
        scored.append((phrase, avg_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ---------------------------
# TITLES & HASHTAGS
# ---------------------------
def generate_titles(phrases, title):
    titles = []
    base_context = title.strip().capitalize()
    if phrases and phrases[0][1] > 0:
        main = phrases[0][0].capitalize()
        titles.extend([
            f"{base_context}: Why {main} Is Getting Massive Attention 🚀",
            f"This {main} Moment Changed Everything 🤯",
            f"People Didn’t Expect This {main} Outcome 😱"
        ])
    else:
        titles.extend([
            f"{base_context}: What Happens Next Will Surprise You 🤯",
            f"This Short Is Simpler Than It Looks — But Powerful 🚀",
            f"Why This Moment Is Getting So Much Attention 👀"
        ])
    return titles[:3]

def generate_hashtags(phrases):
    tags = []
    for phrase, _ in phrases:
        for w in phrase.split():
            tags.append("#" + w)
    tags += ["#shorts", "#youtube", "#viral"]
    return list(dict.fromkeys(tags))[:10]

def viral_score(views):
    return min(100, int(np.log10(views + 1) * 25))

def youtube_links(phrases):
    links = []
    for phrase, _ in phrases[:5]:
        q = phrase.replace(" ", "+")
        links.append(f"https://www.youtube.com/results?search_query={q}+shorts")
    return links

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="YouTube Shorts Virality AI", page_icon="🚀", layout="wide")
st.title("🚀 YouTube Shorts Virality AI")
st.caption("Dataset-driven • Linguistically corrected • Fully local")

with st.sidebar:
    st.header("🎯 Inputs")
    title = st.text_input("Title (English)")
    description = st.text_area("Script / Description")
    hashtags_input = st.text_input("Existing hashtags (optional)")
    duration = st.slider("Duration (seconds)", 5, 60, 30)

if st.button("🔮 Analyze & Predict"):

    full_text = f"{title} {description}"

    # ================= PREDICTION =================
    X_text = tfidf.transform([full_text])
    X_num = scaler.transform([[duration, len(title), len(description), len(hashtags_input.split())]])
    X_user = hstack([X_text, X_num])
    views = int(np.expm1(model.predict(X_user)[0]))
    v_score = viral_score(views)

    # ================= NLP =================
    phrases = extract_key_phrases(full_text)
    scored_phrases = score_phrases_with_dataset(phrases)

    # ========= LAYOUT =========
    col1, col2 = st.columns([2,1])

    with col1:
        st.success(f"📈 Estimated Views: {views:,}")
        st.metric("🔥 Viral Score", f"{v_score}/100")
        with st.expander("🔍 Voir pourquoi"):
            reasons = []
            if phrases:
                reasons.append("Detected keywords with proven high average views in dataset")
            if duration <= 30:
                reasons.append("Short duration fits viral Shorts format")
            if len(description.split()) > 5:
                reasons.append("Rich content increases engagement likelihood")
            if not reasons:
                reasons.append("Fallback to linguistic context")
            for r in reasons:
                st.write("✔️", r)

        st.subheader("📋 Keyword Average Views")
        if scored_phrases:
            for phrase, score in scored_phrases:
                st.write(f"• '{phrase}' → ~{score:,} avg views")
        else:
            st.write("No keywords detected yet.")

    with col2:
        if scored_phrases:
            kw, kw_scores = zip(*scored_phrases)
            fig = px.bar(
                x=kw,
                y=kw_scores,
                text=[f"{v:,}" for v in kw_scores],
                color=kw_scores,
                color_continuous_scale='Blues',
                title="Keyword Dataset Scores"
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

    # ================= TITLES =================
    st.subheader("✍️ Optimized Titles")
    for t in generate_titles(scored_phrases, title):
        st.write("•", t)

    # ================= HASHTAGS =================
    st.subheader("🏷️ Suggested Hashtags")
    st.code(" ".join(generate_hashtags(scored_phrases)))

    # ================= LINKS =================
    st.subheader("🔗 YouTube Inspiration")
    for link in youtube_links(scored_phrases):
        st.write(link)

st.markdown("---")
st.caption("Linguistically-aware ML • Dataset-driven • Streamlit front-end enhanced with graphs")
