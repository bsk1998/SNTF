from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math, re

router = APIRouter()

# ═══════════════════════════════════════════
# SCHÉMAS
# ═══════════════════════════════════════════
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
# EMBEDDING — HuggingFace (sémantique réel)
# ═══════════════════════════════════════════
_hf_model = None

def get_embedding(text: str) -> list:
    """
    Embedding sémantique via HuggingFace API.
    Fallback sur sentence-transformers local, puis hash MD5.
    """
    global _hf_model
    hf_key = os.environ.get("HF_API_KEY", "")

    # 1. HuggingFace API (priorité)
    if hf_key:
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                headers={"Authorization": f"Bearer {hf_key}"},
                json={"inputs": text[:512], "options": {"wait_for_model": True}},
                timeout=20
            )
            if r.status_code == 200:
                emb = r.json()
                if isinstance(emb[0], list):
                    import numpy as np
                    emb = np.mean(emb, axis=0).tolist()
                if len(emb) > 0:
                    return emb
        except Exception as e:
            print(f"[HF API] {e}")

    # 2. sentence-transformers local
    if _hf_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _hf_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except:
            _hf_model = False

    if _hf_model:
        try:
            return _hf_model.encode(text[:512]).tolist()
        except:
            pass

    # 3. Fallback hash MD5
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:200]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x / norm for x in vector]

# ═══════════════════════════════════════════
# RECHERCHE DOCUMENTS
# ═══════════════════════════════════════════
def search_docs(question: str, limit: int = 5) -> list:
    try:
        from database import get_db
        emb = get_embedding(question)
        if not emb:
            return []
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
            if r[0] and float(r[2]) > 0.15:
                try:
                    meta = json.loads(r[1]) if r[1] else {}
                except:
                    meta = {}
                results.append({
                    "content": r[0],
                    "filename": meta.get("filename", ""),
                    "chunk": meta.get("chunk", ""),
                    "score": float(r[2])
                })
        return results
    except Exception as e:
        print(f"[search_docs] {e}")
        return []

# ═══════════════════════════════════════════
# MÉMOIRE CONVERSATION
# ═══════════════════════════════════════════
def ensure_conv_table(cur, conn):
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

def load_history(user_email: str, limit: int = 4) -> list:
    """Charge les N derniers échanges pour la mémoire courte."""
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        ensure_conv_table(cur, conn)
        cur.execute(
            "SELECT question, answer FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT %s",
            (user_email, limit)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        history = []
        for r in reversed(rows):
            history.append({"role": "user", "content": r[0]})
            # Résumé court de la réponse précédente — max 200 chars
            history.append({"role": "assistant", "content": r[1][:200]})
        return history
    except Exception as e:
        print(f"[load_history] {e}")
        return []

def save_conv(user_email: str, question: str, answer: str):
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        ensure_conv_table(cur, conn)
        cur.execute(
            "INSERT INTO conversations (user_email, question, answer) VALUES (%s, %s, %s)",
            (user_email, question[:1000], answer[:3000])
        )
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"[save_conv] {e}")

# ═══════════════════════════════════════════
# DÉTECTION LANGUE
# ═══════════════════════════════════════════
def detect_lang(text: str) -> str:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return "ar" if arabic > len(text) * 0.3 else "fr"

def classify_question(q: str) -> str:
    q_lower = q.lower().strip()

    # Salutations simples
    greetings = ["salut", "bonjour", "bonsoir", "merci", "hello", "hi",
                 "ca va", "salam", "bonne journee", "bonne nuit"]
    if len(q.split()) <= 5 and any(g in q_lower for g in greetings):
        return "greeting"

    # Mots d'urgence / technique
    urgent_words = [
        "panne", "urgent", "urgence", "bloqué", "bloquée", "blocage",
        "alarme", "alerte", "erreur", "code", "défaut", "incident",
        "arrêt", "arrêté", "stopper", "démarrer", "démarrage",
        "redémarrer", "marche pas", "fonctionne pas", "ne répond",
        "comment", "procédure", "étape", "que faire", "quoi faire",
        "عطل", "خطأ", "مشكل", "توقف", "كيف", "إيقاف", "تشغيل"
    ]
    if any(u in q_lower for u in urgent_words):
        return "urgent"

    # Demande de source explicite
    source_words = ["source", "document", "page", "référence", "où", "dans quel"]
    if any(s in q_lower for s in source_words):
        return "source"

    return "question"

# ═══════════════════════════════════════════
# GROQ VISION
# ═══════════════════════════════════════════
def call_groq_vision(question: str, image_b64: str, image_type: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Expert SNTF. Analyse cette image.\n- Lis tout le texte visible (codes erreur, voyants, affichages)\n- Identifie le problème\n- Donne la solution en étapes courtes\nQuestion : " + question},
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_b64}"}}
                ]}],
                "max_tokens": 600
            },
            timeout=60
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return call_groq_text(question)
    except Exception as e:
        return f"⚠️ Erreur vision : {str(e)}"

def call_groq_text(question: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "⚠️ Clé Groq manquante."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": question}],
                  "max_tokens": 600, "temperature": 0.3},
            timeout=30
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"⚠️ Erreur {r.status_code}"
    except Exception as e:
        return f"⚠️ Erreur : {str(e)}"

# ═══════════════════════════════════════════
# HISTORY ENDPOINT
# ═══════════════════════════════════════════
@router.post("/history")
def get_history(req: HistoryRequest):
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        ensure_conv_table(cur, conn)
        cur.execute(
            "SELECT question, answer, created_at FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT 50",
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
# ASK — ENDPOINT PRINCIPAL
# ═══════════════════════════════════════════
@router.post("/ask")
async def ask(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "Question vide")

    key = os.environ.get("GROQ_API_KEY", "")

    # ── Image ──
    if req.image:
        b64 = req.image
        if "base64," in b64:
            b64 = b64.split("base64,")[1]
        answer = call_groq_vision(question, b64, req.image_type or "image/jpeg")
        if req.user_email and req.user_email != "admin":
            save_conv(req.user_email, "[image] " + question, answer)
        return {"answer": answer, "sources": [], "mode": "vision"}

    lang    = detect_lang(question)
    q_type  = classify_question(question)

    # ── Recherche documents ──
    docs = []
    context = ""
    if q_type in ("question", "urgent", "source"):
        docs = search_docs(question, limit=5)
        if docs:
            # Trier par score décroissant — garder les 3 meilleurs
            docs = sorted(docs, key=lambda d: d["score"], reverse=True)[:3]
            parts = []
            for d in docs:
                prefix = f"[{d['filename']}]\n" if d['filename'] else ""
                parts.append(prefix + d['content'][:500])
            context = "\n---\n".join(parts)

    # ── Historique court ──
    history = []
    if req.user_email and req.user_email != "admin":
        history = load_history(req.user_email, limit=4)

    # ══════════════════════════════════════════════════════════════
    # PROMPT selon le type de question
    # ══════════════════════════════════════════════════════════════
    if q_type == "greeting":
        system = "Tu es l'assistant SNTF. Réponds chaleureusement en 1-2 phrases et propose ton aide pour les questions techniques ou procédures SNTF."

    elif q_type == "urgent":
        system = """Assistant terrain SNTF — MODE URGENCE.

RÈGLES STRICTES :
1. Réponds en MAX 6 lignes
2. Si tu as la solution → étapes numérotées immédiates (1. Fais X  2. Appuie sur Y)
3. Si info manquante → UNE seule question : "Quel code erreur ?" ou "Quelle rame ?"
4. Jamais d'introduction. Jamais de "selon les documents". Jamais de paragraphe.
5. Termine par : "✅ Résolu ?" ou "⚠️ Si persiste → Centre de contrôle"
6. Langue : """ + ("arabe" if lang == "ar" else "français")

    elif q_type == "source":
        system = """Assistant SNTF. L'utilisateur demande les sources.
Réponds avec la réponse ET indique clairement : "📄 Source : [nom du fichier], chunk [numéro]"
Sois précis sur l'origine de chaque information."""

    else:
        system = """Assistant terrain SNTF — expert ferroviaire algérien.

RÈGLES :
1. Réponse directe et courte — max 8 lignes
2. Procédure → étapes numérotées courtes
3. Pas d'introduction, pas de "selon les documents"
4. Si info insuffisante → 1 seule question ciblée
5. Langue : """ + ("arabe" if lang == "ar" else "français")

    # ── Construction messages ──
    messages = [{"role": "system", "content": system}]
    messages.extend(history)

    if context:
        user_content = f"[Doc SNTF]: {context}\n\n[Question]: {question}"
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})

    # ── Boutons de suivi (générés selon contexte) ──
    def get_quick_replies(q_type: str, has_docs: bool) -> list:
        if q_type == "greeting":
            return []
        buttons = []
        if q_type == "urgent":
            buttons = ["✅ Résolu", "⚠️ Ça persiste", "📞 Centre de contrôle"]
        elif has_docs:
            buttons = ["📄 Sources du document", "🔍 Plus de détails", "📋 Étape suivante"]
        else:
            buttons = ["🔍 Plus de détails", "📋 Autre question", "📞 Assistance"]
        return buttons

    quick_replies = get_quick_replies(q_type, bool(docs))
    sources = list(set(d["filename"] for d in docs if d["filename"]))

    # ── Streaming ──
    def generate():
        full_answer = ""

        # Premier chunk : métadonnées
        yield "data: " + json.dumps({
            "sources": sources,
            "web": False,
            "quick_replies": quick_replies
        }) + "\n\n"

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
                    "max_tokens": 600,
                    "temperature": 0.3,
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
