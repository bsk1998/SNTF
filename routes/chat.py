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
# 🚀 CACHE MODÈLE
# ═══════════════════════════════════════
_embedding_model = None

def get_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Modèle embedding chargé")
        except Exception as e:
            print(f"sentence-transformers non dispo: {e}")
    return _embedding_model

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
# 🌐 RECHERCHE WEB DUCKDUCKGO (GRATUIT)
# ═══════════════════════════════════════
def search_web(query: str) -> str:
    """Recherche DuckDuckGo — 100% gratuit, sans clé API"""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query + " SNTF Algérie",
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            },
            timeout=8,
            headers={"User-Agent": "SNTF-Assistant/1.0"}
        )
        data = r.json()

        results = []

        # Résumé principal
        if data.get("AbstractText"):
            results.append(data["AbstractText"][:400])

        # Résultats connexes
        for item in data.get("RelatedTopics", [])[:3]:
            if isinstance(item, dict) and item.get("Text"):
                results.append(item["Text"][:200])

        if results:
            return "\n".join(results)
        return ""
    except Exception as e:
        print(f"web_search error: {e}")
        return ""

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
            (user_email, question[:1000], answer[:5000])
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
            "SELECT question, answer, created_at FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT %s",
            (user_email, limit)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        rows.reverse()
        return [{"question": r[0], "answer": r[1], "time": str(r[2]) if r[2] else ''} for r in rows]
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
                "max_tokens": 1500,
                "temperature": 0.4,
                "stream": True
            },
            stream=True,
            timeout=60
        )
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
                            yield "data: " + json.dumps({"token": token}) + "\n\n"
                    except: pass
    except Exception as e:
        yield "data: " + json.dumps({"token": f"Erreur: {e}"}) + "\n\n"
        yield "data: [DONE]\n\n"

def call_groq(messages: list) -> str:
    if not GROQ_API_KEY: return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 1500, "temperature": 0.4},
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
                    {"type": "text", "text": f"Tu es l'assistant SNTF. Analyse cette image en détail et réponds : {question}"},
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
    history = load_history(req.user_email, limit=20)  # réponses complètes
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

    # ── Recherche docs PDF ──
    docs = search_docs(question)
    context = ""
    sources = []
    for d in docs[:3]:
        if d["score"] > 0.25:
            context += d["text"][:500] + "\n\n"
            meta = d.get("metadata", {})
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            fname = meta.get("filename") or meta.get("source", "")
            if fname and fname not in sources:
                sources.append(fname)

    # ── Recherche web si pas assez de contexte PDF ──
    web_context = ""
    if not context or len(context) < 100:
        print(f"🌐 Recherche web pour: {question[:50]}")
        web_context = search_web(question)
        if web_context:
            print(f"🌐 Web résultat: {web_context[:80]}...")

    # ── Historique utilisateur ──
    db_history = []
    if req.user_email and req.user_email != "admin":
        db_history = load_history(req.user_email, limit=6)

    # ═══════════════════════════════════════════════════════════
    # 🧠 PROMPT AMÉLIORÉ — Expert SNTF qui explique et analyse
    # ═══════════════════════════════════════════════════════════
    system = """Tu es SNTF Expert — assistant IA officiel de la Société Nationale des Transports Ferroviaires d'Algérie. Tu agis comme un expert terrain réactif dont le but est de RÉSOUDRE les problèmes rapidement.

═══════════════════════════════════════
PRIORITÉ ABSOLUE : DONNER LA SOLUTION
═══════════════════════════════════════

RÈGLE N°1 — DIAGNOSTIC AVANT TOUT :
Si la question manque d'informations cruciales pour donner une solution précise,
pose UNE SEULE question ciblée avant de répondre. Exemple :
• "Quelle ligne / gare est concernée ?"
• "S'agit-il d'un problème technique ou administratif ?"
• "Quand cela s'est-il produit ?"
Ne pose JAMAIS plusieurs questions en même temps.

RÈGLE N°2 — STRUCTURE DE RÉPONSE OBLIGATOIRE :
Pour chaque réponse, suis ce format :
✅ SOLUTION : [La solution directe et actionnable en 1-2 phrases]
📋 DÉTAILS : [Explication brève si nécessaire — maximum 3 points]
📞 ACTION : [Qui contacter / quoi faire concrètement — si applicable]

RÈGLE N°3 — PRÉCISION ET HONNÊTETÉ :
• Si tu as l'info dans les documents → solution précise basée sur les faits
• Si l'info vient du web → indique-le et donne la solution
• Si tu n'as PAS l'info → dis clairement "Je n'ai pas cette information précise" 
  puis propose une alternative (qui contacter, où chercher)
• Ne JAMAIS inventer des numéros, noms, procédures ou données

RÈGLE N°4 — STYLE :
• Réponses courtes et percutantes — pas de paragraphes interminables
• Commence toujours par la solution, jamais par une explication
• Si la question est en arabe → réponds en arabe avec le même format
• Utilise des emojis pour structurer visuellement (✅ 📋 ⚠️ 📞 🔧)

RÈGLE N°5 — SUIVI :
Termine chaque réponse par une micro-question de suivi si pertinent :
"Ce problème est-il résolu ou avez-vous besoin de plus de détails ?"

Tu as accès aux documents internes SNTF et à des recherches web en temps réel."""

    messages = [{"role": "system", "content": system}]

    # Historique conversation
    for h in db_history[-4:]:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"][:800]})

    # Construire le contexte enrichi
    parts = []
    if context:
        parts.append(f"📄 DOCUMENTS INTERNES SNTF :\n{context}")
    if web_context:
        parts.append(f"🌐 INFORMATIONS WEB COMPLÉMENTAIRES :\n{web_context}")
    parts.append(f"Question de l'utilisateur : {question}")

    messages.append({"role": "user", "content": "\n\n".join(parts)})

    # ── STREAMING ──
    def generate():
        full = ""
        yield "data: " + json.dumps({"sources": sources, "web": bool(web_context)}) + "\n\n"
        for chunk in stream_groq(messages):
            if "DONE" not in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", ""))
                    full += data.get("token", "")
                except: pass
            yield chunk
        if req.user_email and req.user_email != "admin" and full:
            save_conversation(req.user_email, question, full)

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
