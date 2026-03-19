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
# CACHE MODÈLE EMBEDDING
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
except:
    pass

# ═══════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════
def get_embedding(text: str):
    model = get_model()
    if model:
        try:
            return model.encode(text[:500]).tolist()
        except:
            pass
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
        except:
            pass
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:200]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x / norm for x in vector]

# ═══════════════════════════════════════
# RECHERCHE DOCS VECTORIELLE
# ═══════════════════════════════════════
def search_docs(question: str, limit=3):
    try:
        from database import get_db
        emb = get_embedding(question)
        if not emb:
            return []
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
        cur.close()
        conn.close()
        return [{"text": r[0], "metadata": r[1], "score": float(r[2])} for r in rows]
    except Exception as e:
        print(f"search_docs error: {e}")
        return []

# ═══════════════════════════════════════
# RECHERCHE WEB DUCKDUCKGO
# ═══════════════════════════════════════
def search_web(query: str) -> str:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query + " SNTF Algérie", "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
            headers={"User-Agent": "SNTF-Assistant/1.0"}
        )
        data = r.json()
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"][:400])
        for item in data.get("RelatedTopics", [])[:3]:
            if isinstance(item, dict) and item.get("Text"):
                results.append(item["Text"][:200])
        return "\n".join(results) if results else ""
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
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (user_email, question, answer) VALUES (%s, %s, %s)",
            (user_email, question[:1000], answer[:5000])
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"save_conversation error: {e}")

def load_history(user_email: str, limit=6):
    try:
        from database import get_db
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT question, answer, created_at FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT %s",
            (user_email, limit)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        rows.reverse()
        return [{"question": r[0], "answer": r[1], "time": str(r[2]) if r[2] else ""} for r in rows]
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
    images: Optional[list] = None
    pdfs: Optional[list] = None
    memory: Optional[list] = None
    user_email: Optional[str] = None
    stream: Optional[bool] = True

class HistoryRequest(BaseModel):
    user_email: str

# ═══════════════════════════════════════
# GROQ — STREAMING
# ═══════════════════════════════════════
def stream_groq(messages: list):
    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        yield "data: " + json.dumps({"token": "⚠️ Clé Groq manquante — vérifiez GROQ_API_KEY sur Render."}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
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
        if r.status_code != 200:
            print(f"❌ Groq HTTP {r.status_code}: {r.text[:300]}")
            yield "data: " + json.dumps({"token": f"⚠️ Erreur Groq {r.status_code} — vérifiez la clé API sur Render."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        for line in r.iter_lines():
            if line:
                line = line.decode("utf-8")
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
                    except:
                        pass
    except requests.exceptions.Timeout:
        yield "data: " + json.dumps({"token": "⚠️ Timeout — Groq ne répond pas."}) + "\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"stream_groq exception: {e}")
        yield "data: " + json.dumps({"token": f"⚠️ Erreur : {str(e)}"}) + "\n\n"
        yield "data: [DONE]\n\n"

# ═══════════════════════════════════════
# GROQ — NON STREAMING
# ═══════════════════════════════════════
def call_groq(messages: list) -> str:
    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 1500, "temperature": 0.4},
            timeout=60
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        print(f"❌ call_groq HTTP {r.status_code}: {r.text[:200]}")
        return f"⚠️ Erreur Groq ({r.status_code}) — vérifiez la clé API."
    except Exception as e:
        return f"⚠️ Erreur : {e}"

# ═══════════════════════════════════════
# GROQ — TEXTE SIMPLE (PDF sans image)
# ═══════════════════════════════════════
def call_groq_text(question: str) -> str:
    messages = [
        {"role": "system", "content": "Tu es un expert SNTF. Analyse le document et réponds avec précision."},
        {"role": "user", "content": question}
    ]
    return call_groq(messages)

# ═══════════════════════════════════════
# GROQ — VISION 1 IMAGE
# ═══════════════════════════════════════
def call_groq_vision(question: str, image_b64: str, image_type: str) -> str:
    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"Expert SNTF. Analyse l'image : lis tout le texte visible, identifie les problèmes, donne une solution.\nQuestion : {question}"},
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
        return f"⚠️ Erreur vision : {e}"

# ═══════════════════════════════════════
# GROQ — VISION MULTI-IMAGES
# ═══════════════════════════════════════
def call_groq_multi_vision(question: str, imgs: list, pdf_context: str = "") -> str:
    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return "⚠️ Clé Groq manquante."
    try:
        content = [{"type": "text", "text": f"Expert SNTF. Analyse ces {len(imgs)} images.\nQuestion : {question}"}]
        for img in imgs[:4]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.get('type','image/jpeg')};base64,{img['b64']}"}
            })
        if pdf_context:
            content.append({"type": "text", "text": f"\nContexte PDF :{pdf_context[:2000]}"})
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1500
            },
            timeout=90
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return call_groq_vision(question, imgs[0]["b64"], imgs[0].get("type", "image/jpeg"))
    except Exception as e:
        return f"⚠️ Erreur multi-vision : {e}"

# ═══════════════════════════════════════
# ENDPOINT DIAGNOSTIC
# ═══════════════════════════════════════
@router.get("/test")
async def test_groq():
    """Teste la connexion Groq — ouvrir /api/chat/test dans le navigateur."""
    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        return {"status": "❌ ERREUR", "message": "GROQ_API_KEY absente — ajoutez-la dans Render > Environment"}
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Réponds juste: OK"}], "max_tokens": 5},
            timeout=15
        )
        if r.status_code == 200:
            return {"status": "✅ OK", "groq": "connecté", "key_prefix": key[:10] + "..."}
        return {"status": "❌ ERREUR", "http_code": r.status_code, "detail": r.text[:300]}
    except Exception as e:
        return {"status": "❌ ERREUR", "exception": str(e)}

# ═══════════════════════════════════════
# ENDPOINTS PRINCIPAUX
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

    has_images = (req.images and len(req.images) > 0) or req.image
    has_pdfs   = req.pdfs and len(req.pdfs) > 0

    if has_images or has_pdfs:
        pdf_context = ""
        if has_pdfs:
            import base64 as b64lib, io
            try:
                from pypdf import PdfReader
                for pdf in req.pdfs:
                    try:
                        pdf_bytes = b64lib.b64decode(pdf["b64"])
                        reader    = PdfReader(io.BytesIO(pdf_bytes))
                        text = "".join(page.extract_text() or "" for page in reader.pages)
                        pdf_context += f"\n\n=== PDF: {pdf['name']} ===\n{text[:3000]}"
                    except Exception as e:
                        pdf_context += f"\n[Erreur PDF {pdf['name']}: {e}]"
            except:
                pass

        if has_images:
            imgs = list(req.images or [])
            if req.image:
                imgs = [{"b64": req.image, "type": req.image_type or "image/jpeg", "name": "image"}] + imgs
            if pdf_context:
                question = f"{question}\n\nContexte PDFs:{pdf_context}"
            answer = call_groq_multi_vision(question, imgs, pdf_context) if len(imgs) > 1 \
                     else call_groq_vision(question, imgs[0]["b64"], imgs[0].get("type", "image/jpeg"))
        else:
            answer = call_groq_text(f"{question}\n\nDocuments:{pdf_context}")

        if req.user_email and req.user_email != "admin":
            save_conversation(req.user_email, f"[fichiers] {question[:200]}", answer)
        return {"answer": answer, "sources": [], "mode": "multifile"}

    # ── Recherche docs ──
    docs    = search_docs(question)
    context = ""
    sources = []
    for d in docs[:3]:
        if d["score"] > 0.25:
            context += d["text"][:500] + "\n\n"
            meta = d.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            fname = meta.get("filename") or meta.get("source", "")
            if fname and fname not in sources:
                sources.append(fname)

    web_context = ""
    if not context or len(context) < 100:
        web_context = search_web(question)

    db_history = []
    if req.user_email and req.user_email != "admin":
        db_history = load_history(req.user_email, limit=6)

    system = """Tu es SNTF Expert — assistant IA officiel de la Société Nationale des Transports Ferroviaires d'Algérie.

FORMAT OBLIGATOIRE :
✅ SOLUTION : [réponse directe en 1-2 phrases]
📋 DÉTAILS : [max 3 points si nécessaire]
📞 ACTION : [qui contacter / quoi faire — si applicable]

RÈGLES :
• Commence toujours par la solution
• Ne jamais inventer de données
• Si question en arabe → réponds en arabe
• Réponses courtes et percutantes"""

    messages = [{"role": "system", "content": system}]
    for h in db_history[-4:]:
        messages.append({"role": "user",      "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"][:600]})

    parts = []
    if context:
        parts.append(f"📄 DOCUMENTS SNTF :\n{context}")
    if web_context:
        parts.append(f"🌐 WEB :\n{web_context}")
    parts.append(f"Question : {question}")
    messages.append({"role": "user", "content": "\n\n".join(parts)})

    def generate():
        full = ""
        yield "data: " + json.dumps({"sources": sources, "web": bool(web_context)}) + "\n\n"
        for chunk in stream_groq(messages):
            if "DONE" not in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", ""))
                    full += data.get("token", "")
                except:
                    pass
            yield chunk
        if req.user_email and req.user_email != "admin" and full:
            save_conversation(req.user_email, question, full)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
