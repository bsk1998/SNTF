from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY   = os.environ.get("HF_API_KEY")
VECTOR_TABLE = "n8n_vectors"

# ═══════════════════════════════════════
# 🚀 CACHE MODÈLE — chargé une seule fois
# ═══════════════════════════════════════
_embedding_model = None

def get_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("⚙️ Chargement modèle embedding...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Modèle chargé et mis en cache !")
        except Exception as e:
            print(f"sentence-transformers non dispo: {e}")
    return _embedding_model

# Précharger au démarrage du serveur
try:
    get_model()
except: pass

# ═══════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════
def get_embedding(text: str):
    model = get_model()
    if model:
        try:
            return model.encode(text[:500]).tolist()
        except: pass
    if HF_API_KEY:
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": text[:300], "options": {"wait_for_model": True}},
                timeout=30
            )
            if r.status_code == 200:
                emb = r.json()
                if isinstance(emb[0], list):
                    import numpy as np
                    emb = np.mean(emb, axis=0).tolist()
                return emb
        except: pass
    # Fallback rapide
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:200]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x/norm for x in vector]

# ═══════════════════════════════════════
# RECHERCHE DOCS
# ═══════════════════════════════════════
def search_docs(question: str, limit=3):
    try:
        from database import get_db
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

# ═══════════════════════════════════════
# MÉMOIRE SUPABASE
# ═══════════════════════════════════════
def save_conversation(user_email: str, question: str, answer: str):
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (user_email, question, answer) VALUES (%s, %s, %s)",
            (user_email, question[:500], answer[:2000])
        )
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"save_conversation error: {e}")

def load_history(user_email: str, limit=6):
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT question, answer FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT %s",
            (user_email, limit)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        rows.reverse()
        return [{"question": r[0], "answer": r[1]} for r in rows]
    except Exception as e:
        print(f"load_history error: {e}")
        return []

# ═══════════════════════════════════════
# SCHÉMAS
# ═══════════════════════════════════════
class ChatRequest(BaseModel):
    question: str
    image: Optional[str] = None
    image_type: Optional[str] = None
    user_email: Optional[str] = None
    stream: Optional[bool] = True

class HistoryRequest(BaseModel):
    user_email: str

# ═══════════════════════════════════════
# STREAMING GROQ
# ═══════════════════════════════════════
def stream_groq(messages: list):
    """Générateur qui streame token par token depuis Groq"""
    if not GROQ_API_KEY:
        yield "data: " + json.dumps({"token": "⚠️ Clé Groq manquante."}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.3,
                "stream": True  # ← STREAMING ACTIVÉ
            },
            stream=True,
            timeout=60
        )
        full_answer = ""
        for line in r.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            full_answer += token
                            yield "data: " + json.dumps({"token": token}) + "\n\n"
                    except: pass
    except Exception as e:
        yield "data: " + json.dumps({"token": f"Erreur: {e}"}) + "\n\n"
        yield "data: [DONE]\n\n"

# Groq normal (sans stream) pour vision
def call_groq(messages: list) -> str:
    if not GROQ_API_KEY: return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 1024, "temperature": 0.3},
            timeout=60
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"Erreur {r.status_code}"
    except Exception as e:
        return f"Erreur: {e}"

def call_groq_vision(question: str, image_b64: str, image_type: str) -> str:
    if not GROQ_API_KEY: return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.2-90b-vision-preview",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"Tu es l'assistant intelligent de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie). Analyse cette image et réponds en français à la question suivante : {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_b64}"}}
                ]}],
                "max_tokens": 1024
            },
            timeout=60
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return call_groq([{"role": "user", "content": question}])
    except Exception as e:
        return f"Erreur vision: {e}"

# ═══════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════
@router.post("/history")
async def get_history(req: HistoryRequest):
    history = load_history(req.user_email, limit=20)
    return {"success": True, "history": history, "count": len(history)}

@router.post("/ask")
async def ask(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question vide")

    # ── Image ──
    if req.image:
        answer = call_groq_vision(question, req.image, req.image_type or "image/jpeg")
        if req.user_email and req.user_email != "admin":
            save_conversation(req.user_email, f"[IMAGE] {question}", answer)
        return {"answer": answer, "sources": [], "mode": "vision"}

    # ── Recherche docs en parallèle ──
    docs = search_docs(question)
    context = ""
    sources = []
    for d in docs[:3]:
        if d["score"] > 0.3:  # seuil qualité
            context += d["text"][:500] + "\n\n"
            meta = d.get("metadata", {})
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            fname = meta.get("filename") or meta.get("source", "")
            if fname and fname not in sources:
                sources.append(fname)

    # ── Historique court (6 échanges max pour vitesse) ──
    db_history = []
    if req.user_email and req.user_email != "admin":
        db_history = load_history(req.user_email, limit=6)

    # ── Construire messages ──
    system = """Tu es l'assistant SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Réponds en français, de façon concise et professionnelle.
Si la question est en arabe, réponds en arabe.
Utilise le contexte documentaire si disponible."""

    messages = [{"role": "system", "content": system}]

    for h in db_history[-4:]:  # 4 derniers échanges seulement
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"][:300]})

    user_content = question
    if context:
        user_content = f"Contexte:\n{context}\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    # ── STREAMING ──
    def generate():
        full = ""
        # Envoyer sources en premier
        yield "data: " + json.dumps({"sources": sources}) + "\n\n"
        for chunk in stream_groq(messages):
            if "DONE" not in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", ""))
                    full += data.get("token", "")
                except: pass
            yield chunk
        # Sauvegarder après stream complet
        if req.user_email and req.user_email != "admin" and full:
            save_conversation(req.user_email, question, full)

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
