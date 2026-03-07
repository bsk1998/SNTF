from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math
from database import get_db

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY   = os.environ.get("HF_API_KEY")
VECTOR_TABLE = "n8n_vectors"

# ─── Embedding ───
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
                    import numpy as np
                    emb = np.mean(emb, axis=0).tolist()
                return emb
        except: pass
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

# ─── Sauvegarder conversation en base ───
def save_conversation(user_email: str, question: str, answer: str):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (user_email, question, answer) VALUES (%s, %s, %s)",
            (user_email, question, answer[:2000])
        )
        conn.commit()
        cur.close(); conn.close()
        print(f"💾 Conversation sauvegardée pour {user_email}")
    except Exception as e:
        print(f"save_conversation error: {e}")

# ─── Charger historique utilisateur ───
def load_history(user_email: str, limit=20):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """SELECT question, answer, created_at 
               FROM conversations 
               WHERE user_email = %s 
               ORDER BY created_at DESC 
               LIMIT %s""",
            (user_email, limit)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        # Inverser pour ordre chronologique
        rows.reverse()
        return [{"question": r[0], "answer": r[1], "time": str(r[2])} for r in rows]
    except Exception as e:
        print(f"load_history error: {e}")
        return []

# ─── Appel Groq ───
def call_groq(messages: list) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 1024, "temperature": 0.3},
            timeout=60
        )
        data = r.json()
        if r.status_code == 200:
            return data["choices"][0]["message"]["content"]
        return f"Erreur Groq {r.status_code}: {data}"
    except Exception as e:
        return f"Erreur: {e}"

# ─── Groq Vision ───
def call_groq_vision(question: str, image_b64: str, image_type: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.2-90b-vision-preview",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_b64}"}},
                    {"type": "text", "text": f"Tu es l'assistant SNTF. {question}"}
                ]}],
                "max_tokens": 1024
            },
            timeout=60
        )
        data = r.json()
        if r.status_code == 200:
            return data["choices"][0]["message"]["content"]
        return call_groq([{"role": "user", "content": question}])
    except Exception as e:
        return f"Erreur vision: {e}"

# ─── Schémas ───
class ChatRequest(BaseModel):
    question: str
    image: Optional[str] = None
    image_type: Optional[str] = None
    memory: Optional[str] = None
    user_email: Optional[str] = None  # pour mémoire persistante

class HistoryRequest(BaseModel):
    user_email: str

# ─── Endpoint historique ───
@router.post("/history")
async def get_history(req: HistoryRequest):
    if not req.user_email:
        raise HTTPException(400, "Email requis")
    history = load_history(req.user_email, limit=20)
    return {"success": True, "history": history, "count": len(history)}

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
        if req.user_email:
            save_conversation(req.user_email, f"[IMAGE] {question}", answer)
        return {"answer": answer, "sources": [], "mode": "vision"}

    # ── Recherche docs ──
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

    # ── Charger historique persistant depuis Supabase ──
    db_history = []
    if req.user_email and req.user_email != "admin":
        db_history = load_history(req.user_email, limit=10)

    # ── Construire messages ──
    system = """Tu es l'assistant intelligent officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Tu réponds en français de manière professionnelle, précise et utile.
Si tu as du contexte documentaire, utilise-le en priorité.
Si la question est en arabe, réponds en arabe.
Tu te souviens de toute la conversation avec l'utilisateur."""

    messages = [{"role": "system", "content": system}]

    # Ajouter historique Supabase comme contexte de conversation
    if db_history:
        for h in db_history[-8:]:  # 8 derniers échanges
            messages.append({"role": "user", "content": h["question"]})
            messages.append({"role": "assistant", "content": h["answer"][:500]})

    # Message actuel avec contexte PDF
    user_content = question
    if context:
        user_content = f"Contexte documentaire SNTF:\n{context}\n\nQuestion: {question}"

    messages.append({"role": "user", "content": user_content})

    answer = call_groq(messages)

    # ── Sauvegarder en base ──
    if req.user_email and req.user_email != "admin":
        save_conversation(req.user_email, question, answer)

    return {"answer": answer, "sources": sources, "mode": "rag", "history_used": len(db_history)}
