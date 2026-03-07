from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, requests, json, base64
from database import get_db

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY   = os.environ.get("HF_API_KEY")
VECTOR_TABLE = "n8n_vectors"

# ─── Embedding identique à documents.py ───
def get_embedding(text: str):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text).tolist()
    except: pass
    if HF_API_KEY:
        try:
            import numpy as np
            r = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": text[:500], "options": {"wait_for_model": True}},
                timeout=60
            )
            if r.status_code == 200:
                emb = r.json()
                if isinstance(emb[0], list):
                    emb = np.mean(emb, axis=0).tolist()
                return emb
        except: pass
    # Fallback
    import hashlib, math
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:384]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x/norm for x in vector]

# ─── Recherche vectorielle ───
def search_docs(question: str, limit=4):
    try:
        emb = get_embedding(question)
        if not emb: return []
        emb_str = "[" + ",".join(map(str, emb)) + "]"
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(f"""
            SELECT text, metadata, 1-(embedding <=> %s::vector) AS score
            FROM {VECTOR_TABLE}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (emb_str, emb_str, limit))
        rows = cur.fetchall()
        cur.close(); conn.close()
        return [{"text": r[0], "metadata": r[1], "score": float(r[2])} for r in rows]
    except Exception as e:
        print(f"search_docs error: {e}")
        return []

# ─── Appel Groq (texte seul ou avec image via vision) ───
def call_groq(messages: list) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.3
            },
            timeout=60
        )
        data = r.json()
        if r.status_code == 200:
            return data["choices"][0]["message"]["content"]
        return f"Erreur Groq {r.status_code}: {data}"
    except Exception as e:
        return f"Erreur: {e}"

# ─── Groq Vision pour images ───
def call_groq_vision(question: str, image_b64: str, image_type: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.2-90b-vision-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Tu es l'assistant intelligent de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie). {question}"
                        }
                    ]
                }],
                "max_tokens": 1024
            },
            timeout=60
        )
        data = r.json()
        if r.status_code == 200:
            return data["choices"][0]["message"]["content"]
        print(f"Groq vision error: {data}")
        # Fallback texte si vision échoue
        return call_groq([{
            "role": "user",
            "content": f"L'utilisateur a envoyé une image (modèle vision indisponible). Question: {question}\n\nRéponds que tu as bien reçu l'image mais que l'analyse visuelle nécessite une connexion au modèle vision."
        }])
    except Exception as e:
        return f"Erreur vision: {e}"

# ─── Schémas ───
class ChatRequest(BaseModel):
    question: str
    image: Optional[str] = None
    image_type: Optional[str] = None
    memory: Optional[str] = None  # contexte conversation

# ─── Endpoint principal ───
@router.post("/ask")
async def ask(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question vide")

    # ── Cas avec image ──
    if req.image:
        print(f"🖼️ Image reçue ({req.image_type}), question: {question[:60]}")
        answer = call_groq_vision(question, req.image, req.image_type or "image/jpeg")
        return {"answer": answer, "sources": [], "mode": "vision"}

    # ── Cas texte seul ──
    docs = search_docs(question)
    context = ""
    sources = []
    if docs:
        for d in docs[:3]:
            context += d["text"][:600] + "\n\n"
            meta = d.get("metadata", {})
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            fname = meta.get("filename") or meta.get("source", "")
            if fname and fname not in sources:
                sources.append(fname)

    system = """Tu es l'assistant intelligent officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Tu réponds en français de manière professionnelle, précise et utile.
Si tu as du contexte documentaire, utilise-le en priorité.
Si tu détectes que la question est en arabe, réponds en arabe.
Tu te souviens du contexte de la conversation si fourni."""

    user_content = question
    parts = []
    if req.memory:
        parts.append(f"Historique conversation:\n{req.memory}")
    if context:
        parts.append(f"Contexte documentaire SNTF:\n{context}")
    parts.append(f"Question: {question}")
    user_content = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]

    answer = call_groq(messages)
    return {"answer": answer, "sources": sources, "mode": "rag"}
