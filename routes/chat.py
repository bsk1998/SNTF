from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math, re

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    image: Optional[str] = None
    image_type: Optional[str] = None
    images: Optional[list] = None
    pdfs: Optional[list] = None
    memory: Optional[object] = None
    user_email: Optional[str] = None
    stream: Optional[bool] = True

class HistoryRequest(BaseModel):
    user_email: str

# ═══════════════════════════════════════════
# DIAGNOSTIC
# ═══════════════════════════════════════════
@router.get("/test")
def test_all():
    results = {}
    key = os.environ.get("GROQ_API_KEY", "")
    results["groq_key_present"] = bool(key)
    results["groq_key_prefix"] = key[:10] + "..." if key else "ABSENTE"
    if key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": "dis OK"}],
                      "max_tokens": 5},
                timeout=15
            )
            results["groq_status"] = r.status_code
            if r.status_code == 200:
                results["groq_response"] = r.json()["choices"][0]["message"]["content"]
            else:
                results["groq_error"] = r.text[:200]
        except Exception as e:
            results["groq_exception"] = str(e)
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        results["document_chunks_count"] = cur.fetchone()[0]
        cur.execute("SELECT to_regclass('public.conversations')")
        results["conversations_table"] = "existe" if cur.fetchone()[0] else "ABSENTE"
        cur.close(); conn.close()
        results["supabase"] = "connecté"
    except Exception as e:
        results["supabase_error"] = str(e)
    return results

# ═══════════════════════════════════════════
# HISTORY ENDPOINT
# ═══════════════════════════════════════════
@router.post("/history")
def get_history(req: HistoryRequest):
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT question, answer, created_at FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT 20",
            (req.user_email,)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        rows.reverse()
        return {"success": True, "history": [
            {"question": r[0], "answer": r[1], "time": str(r[2] or "")} for r in rows
        ], "count": len(rows)}
    except Exception as e:
        return {"success": False, "history": [], "count": 0, "error": str(e)}

# ═══════════════════════════════════════════
# RECHERCHE DOCUMENTS
# ═══════════════════════════════════════════
def search_docs(question: str, limit: int = 6) -> list:
    try:
        from database import get_db
        words = question.lower().split()
        vector = [0.0] * 384
        for i, word in enumerate(words[:200]):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[h % 384] += 1.0 / (i + 1)
        norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
        emb = [x / norm for x in vector]
        emb_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT content, metadata, 1-(embedding <=> %s::vector) AS score FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (emb_str, emb_str, limit)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        results = []
        for r in rows:
            if r[0] and float(r[2]) > 0.1:
                try:
                    meta = json.loads(r[1]) if r[1] else {}
                except:
                    meta = {}
                results.append({
                    "content": r[0],
                    "filename": meta.get("filename", ""),
                    "score": float(r[2])
                })
        return results
    except Exception as e:
        print(f"[search_docs] {e}")
        return []

# ═══════════════════════════════════════════
# MÉMOIRE CONVERSATION
# ═══════════════════════════════════════════
def load_history(user_email: str, limit: int = 8) -> list:
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_email TEXT,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.execute(
            "SELECT question, answer FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT %s",
            (user_email, limit)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        history = []
        for r in reversed(rows):
            history.append({"role": "user",      "content": r[0]})
            history.append({"role": "assistant",  "content": r[1][:800]})
        return history
    except Exception as e:
        print(f"[load_history] {e}")
        return []

def save_conv(user_email: str, question: str, answer: str):
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
        print(f"[save_conv] {e}")

# ═══════════════════════════════════════════
# DÉTECTION LANGUE ET TYPE DE QUESTION
# ═══════════════════════════════════════════
def detect_lang(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars > len(text) * 0.3 else "fr"

def classify_question(q: str) -> str:
    q_lower = q.lower().strip()
    greetings = ["salut", "bonjour", "bonsoir", "merci", "hello", "hi", "ça va",
                 "ca va", "salam", "مرحبا", "السلام", "شكرا", "أهلا", "كيف", "bonne journée"]
    if len(q.split()) <= 5 and any(g in q_lower for g in greetings):
        return "greeting"
    if "?" not in q and len(q.split()) <= 3:
        return "short"
    return "question"

# ═══════════════════════════════════════════
# ASK — ENDPOINT PRINCIPAL
# ═══════════════════════════════════════════
@router.post("/ask")
async def ask(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "Question vide")

    key = os.environ.get("GROQ_API_KEY", "")
    lang = detect_lang(question)
    q_type = classify_question(question)

    # ── Recherche documents (sauf salutations) ──
    docs = []
    context = ""
    if q_type == "question":
        docs = search_docs(question, limit=6)
        if docs:
            parts = []
            for d in docs:
                prefix = f"[{d['filename']}]\n" if d['filename'] else ""
                parts.append(prefix + d['content'])
            context = "\n\n---\n\n".join(parts)

    # ── Historique conversation ──
    history = []
    if req.user_email and req.user_email != "admin":
        history = load_history(req.user_email, limit=8)

    # ══════════════════════════════════════════════════════════════
    # PROMPT — Conçu pour des réponses intelligentes et naturelles
    # ══════════════════════════════════════════════════════════════
    system = """Tu es l'assistant IA de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Tu as la personnalité d'un expert ferroviaire algérien : compétent, direct, chaleureux.

## COMMENT TU RÉPONDS

**Analyse d'abord** : Avant de répondre, identifie exactement ce que l'utilisateur veut savoir.
- S'il pose une question précise → donne une réponse précise et directe
- S'il décrit un problème → propose une solution concrète
- S'il dit bonjour → réponds naturellement sans sur-expliquer

**Format adapté** :
- Question simple → 1 à 3 phrases maximum, pas de liste
- Question technique → structure claire avec étapes numérotées si nécessaire
- Procédure → étapes courtes et numérotées
- Ne jamais commencer par "Bien sûr !", "Certainement !", "Absolument !" — va droit au but

**Utilise les documents intelligemment** :
- Si un document répond → cite l'information précise, pas tout le document
- Si plusieurs documents répondent → synthétise, ne répète pas
- Si aucun document ne répond → dis-le en une phrase et réponds avec tes connaissances

**Langue** :
- Français → réponds en français
- Arabe → réponds en arabe
- Mélange arabe/français → réponds dans les deux, arabe en premier

**Ce que tu ne fais PAS** :
- Répéter la question avant de répondre
- Écrire des introductions inutiles
- Lister des informations non demandées
- Dire "selon les documents fournis" à chaque phrase
- Faire des réponses de plus de 300 mots sauf si vraiment nécessaire"""

    # ── Construire les messages ──
    messages = [{"role": "system", "content": system}]
    messages.extend(history)  # mémoire des échanges précédents

    if context:
        user_content = f"Contexte documentaire SNTF :\n{context}\n\n---\n\n{question}"
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})

    # ── Streaming ──
    def generate():
        full_answer = ""
        sources = list(set(d['filename'] for d in docs if d['filename']))
        yield "data: " + json.dumps({"sources": sources, "web": False}) + "\n\n"

        if not key:
            yield "data: " + json.dumps({"token": "⚠️ GROQ_API_KEY manquante."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "max_tokens": 1200,
                    "temperature": 0.4,
                    "stream": True
                },
                stream=True,
                timeout=45
            )

            if r.status_code != 200:
                err = f"⚠️ Erreur Groq {r.status_code}"
                print(f"[groq] {r.text[:200]}")
                yield "data: " + json.dumps({"token": err}) + "\n\n"
                yield "data: [DONE]\n\n"
                return

            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="ignore")
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        full_answer += token
                        yield "data: " + json.dumps({"token": token}) + "\n\n"
                except:
                    pass

        except requests.exceptions.Timeout:
            yield "data: " + json.dumps({"token": "⚠️ Timeout — réessayez."}) + "\n\n"
        except Exception as e:
            print(f"[stream] {e}")
            yield "data: " + json.dumps({"token": f"⚠️ Erreur: {str(e)}"}) + "\n\n"

        yield "data: [DONE]\n\n"

        if full_answer and req.user_email and req.user_email != "admin":
            save_conv(req.user_email, question, full_answer)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
