import os
import json
import pickle
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from config import OPENAI_API_KEY, OPENAI_MODEL

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI()
    except Exception as e:
        print("[warn] OpenAI not available:", e)
        OPENAI_API_KEY = False

# --- Embedding model ---
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "faq_index.pkl")
FAQ_CSV_PATH = os.getenv("FAQ_CSV_PATH", "faqs.csv")

TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SIM = float(os.getenv("MIN_SIM", "0.35"))  # below this → "I don't know"

app = Flask(__name__)

# ------------------------
# Helpers
# ------------------------
def load_faqs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Support flexible column names
    cols = {c.lower(): c for c in df.columns}
    q_col = cols.get("question") or cols.get("questions") or list(df.columns)[0]
    a_col = cols.get("answer") or cols.get("answers") or list(df.columns)[1]
    df = df.rename(columns={q_col: "question", a_col: "answer"})
    # Normalize
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    return df[["question", "answer"]]


def build_or_load_index(
    df: pd.DataFrame,
    embed_model_name: str,
    cache_path: str
) -> Tuple[SentenceTransformer, np.ndarray, List[str], List[str]]:
    model = SentenceTransformer(embed_model_name)

    def save_cache(embeddings: np.ndarray):
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "model": embed_model_name,
                    "embeddings": embeddings,
                    "questions": df["question"].tolist(),
                    "answers": df["answer"].tolist(),
                    "timestamp": time.time(),
                },
                f,
            )

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("model") == embed_model_name and len(blob.get("questions", [])) == len(df):
                # If FAQ count matches, reuse cache
                questions = blob["questions"]
                if questions == df["question"].tolist():
                    return model, blob["embeddings"], blob["questions"], blob["answers"]
        except Exception as e:
            print("[warn] Failed to load cache:", e)

    # Rebuild embeddings
    print("[info] Building FAQ embeddings...")
    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    save_cache(embeddings)
    return model, embeddings, questions, answers


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a and b must be L2-normalized for dot-product == cosine
    return float(np.dot(a, b))


def retrieve_top_k(
    query: str,
    model: SentenceTransformer,
    faq_embeddings: np.ndarray,
    questions: List[str],
    answers: List[str],
    k: int = 3
) -> List[Dict]:
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_vec  # vectorized cosine (since normalized)
    idxs = np.argsort(-sims)[:k]
    results = []
    for i in idxs:
        results.append(
            {
                "question": questions[i],
                "answer": answers[i],
                "score": float(sims[i]),
            }
        )
    return results


def synthesize_answer_direct(query: str, retrieved: List[Dict]) -> str:
    """
    Non-LLM fallback: return the top answer if confident,
    else cite best two answers to be safe.
    """
    if not retrieved:
        return "Sorry, I couldn’t find an answer to that in the HR FAQs."

    top = retrieved[0]
    if top["score"] >= MIN_SIM:
        return top["answer"]

    # Low confidence → gentle fallback
    msg = [
        "I’m not fully sure based on the FAQs, but the closest matches say:",
    ]
    for r in retrieved[:2]:
        msg.append(f"- {r['answer']} (related: “{r['question']}”)")
    msg.append("If this doesn’t help, please rephrase your question or contact HR.")
    return "\n".join(msg)


def synthesize_answer_openai(query: str, retrieved: List[Dict]) -> str:
    """
    LLM synthesis grounded ONLY in retrieved FAQs.
    """
    if not retrieved:
        return "Sorry, I couldn’t find an answer to that in the HR FAQs."

    # Build context
    context_lines = []
    for i, r in enumerate(retrieved, start=1):
        context_lines.append(f"{i}. Q: {r['question']}\n   A: {r['answer']}")
    context = "\n".join(context_lines)

    system_msg = (
        "You are an HR assistant. Answer ONLY using the provided FAQs. "
        "If the answer is not present, say you don't know and suggest contacting HR. "
        "Be concise and friendly. Keep policies factual."
    )
    user_msg = (
        f"User question: {query}\n\n"
        f"Relevant FAQs:\n{context}\n\n"
        f"Rules:\n- Do not invent or assume policies.\n- If unclear, ask the user to clarify."
    )

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[warn] OpenAI generation failed, fallback to direct:", e)
        return synthesize_answer_direct(query, retrieved)


# App init

faq_df = load_faqs(FAQ_CSV_PATH)
embed_model, faq_embeddings, faq_questions, faq_answers = build_or_load_index(
    faq_df, EMBED_MODEL_NAME, EMBED_CACHE_PATH
)


# Routes

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    retrieved = retrieve_top_k(query, embed_model, faq_embeddings, faq_questions, faq_answers, TOP_K)

    if OPENAI_API_KEY:
        answer = synthesize_answer_openai(query, retrieved)
    else:
        answer = synthesize_answer_direct(query, retrieved)

    # mask scores to 3 decimals for display
    for r in retrieved:
        r["score"] = round(r["score"], 3)

    return jsonify(
        {
            "answer": answer,
            "matches": retrieved,
        }
    )

@app.route("/reindex", methods=["POST"])
def reindex():
    """
    Rebuild embeddings if you updated faqs.csv on disk.
    """
    global faq_df, embed_model, faq_embeddings, faq_questions, faq_answers
    faq_df = load_faqs(FAQ_CSV_PATH)
    embed_model, faq_embeddings, faq_questions, faq_answers = build_or_load_index(
        faq_df, EMBED_MODEL_NAME, EMBED_CACHE_PATH
    )
    return jsonify({"status": "ok", "faq_count": len(faq_df)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
