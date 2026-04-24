from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math, time, threading
from collections import defaultdict

router = APIRouter()

# ═══════════════════════════════════════════════════════════
# RATE LIMITING — compteur en mémoire par IP
# Limite : 30 requêtes / 60 secondes / IP
# ═══════════════════════════════════════════════════════════
_rate_store: dict = defaultdict(list)
_rate_lock = threading.Lock()

RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW   = 60

def check_rate_limit(ip: str) -> None:
    now = time.time()
    with _rate_lock:
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_store[ip]) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Trop de requêtes. Limite : {RATE_LIMIT_REQUESTS} messages/minute."
            )
        _rate_store[ip].append(now)

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ═══════════════════════════════════════════════════════════
# SCHÉMAS
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# EMBEDDING — _hf_model protégé par un Lock threading
#
# CORRECTION : le state global _hf_model n'était pas thread-safe.
# Sans lock, deux requêtes simultanées pouvaient corrompre le
# chargement du modèle (double initialisation, accès concurrent).
# Le Lock garantit qu'un seul thread charge le modèle à la fois.
# ═══════════════════════════════════════════════════════════
_hf_model = None
_hf_model_lock = threading.Lock()  # ← CORRECTION : lock ajouté


def get_embedding(text: str) -> list:
    global _hf_model
    hf_key = os.environ.get("HF_API_KEY", "")

    # Méthode 1 : HuggingFace API
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

    # Méthode 2 : modèle local — chargement protégé par Lock
    with _hf_model_lock:
        if _hf_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _hf_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                _hf_model = False

    if _hf_model:
        try:
            return _hf_model.encode(text[:512]).tolist()
        except Exception:
            pass

    # Méthode 3 : fallback hash MD5
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:200]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x / norm for x in vector]


# ═══════════════════════════════════════════════════════════
# RECHERCHE DOCUMENTS — async avec asyncpg
# ═══════════════════════════════════════════════════════════
async def search_docs(question: str, limit: int = 5) -> list:
    try:
        from database import get_pool
        emb = get_embedding(question)
        if not emb:
            return []
        emb_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT content, metadata,
                   1 - (embedding <=> $1::vector) AS score
                   FROM document_chunks
                   ORDER BY embedding <=> $1::vector
                   LIMIT $2""",
                emb_str, limit
            )
        results = []
        for r in rows:
            if r["content"] and float(r["score"]) > 0.15:
                try:
                    meta = json.loads(r["metadata"]) if r["metadata"] else {}
                except Exception:
                    meta = {}
                results.append({
                    "content": r["content"],
                    "filename": meta.get("filename", ""),
                    "chunk": meta.get("chunk", ""),
                    "score": float(r["score"])
                })
        return results
    except Exception as e:
        print(f"[search_docs] {e}")
        return []


# ═══════════════════════════════════════════════════════════
# MÉMOIRE CONVERSATION — async avec asyncpg
#
# CORRECTION : ensure_conv_table() retiré d'ici.
# La table est créée UNE SEULE FOIS au startup dans main.py.
# Avant : appelé à chaque message → 1 requête SQL inutile/message.
# ═══════════════════════════════════════════════════════════
async def load_history(user_email: str, limit: int = 4) -> list:
    try:
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT question, answer
                   FROM conversations
                   WHERE user_email = $1
                   ORDER BY created_at DESC
                   LIMIT $2""",
                user_email, limit
            )
        history = []
        for r in reversed(rows):
            history.append({"role": "user",      "content": r["question"]})
            history.append({"role": "assistant", "content": r["answer"][:200]})
        return history
    except Exception as e:
        print(f"[load_history] {e}")
        return []


async def save_conv(user_email: str, question: str, answer: str):
    try:
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO conversations (user_email, question, answer)
                   VALUES ($1, $2, $3)""",
                user_email, question[:1000], answer[:3000]
            )
    except Exception as e:
        print(f"[save_conv] {e}")


# ═══════════════════════════════════════════════════════════
# DÉTECTION LANGUE ET TYPE DE QUESTION
# ═══════════════════════════════════════════════════════════
def detect_lang(text: str) -> str:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return "ar" if arabic > len(text) * 0.3 else "fr"

def classify_question(q: str) -> str:
    q_lower = q.lower().strip()
    greetings = ["salut", "bonjour", "bonsoir", "merci", "hello", "hi",
                 "ca va", "salam", "bonne journee", "bonne nuit"]
    if len(q.split()) <= 5 and any(g in q_lower for g in greetings):
        return "greeting"
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
    source_words = ["source", "document", "page", "référence", "où", "dans quel"]
    if any(s in q_lower for s in source_words):
        return "source"
    return "question"


# ═══════════════════════════════════════════════════════════
# GROQ VISION
# ═══════════════════════════════════════════════════════════
def call_groq_vision(question: str, image_b64: str, image_type: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "Clé Groq manquante."
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
        return f"Erreur vision : {str(e)}"

def call_groq_text(question: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "Clé Groq manquante."
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
        return f"Erreur {r.status_code}"
    except Exception as e:
        return f"Erreur : {str(e)}"


# ═══════════════════════════════════════════════════════════
# DIAGNOSTIC
# ═══════════════════════════════════════════════════════════
@router.get("/test")
async def test_all():
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
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            results["document_chunks_count"] = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
            conv_exists = await conn.fetchval("SELECT to_regclass('public.conversations')")
            results["conversations_table"] = "existe" if conv_exists else "ABSENTE"
        results["supabase"] = "connecté (asyncpg pool)"
    except Exception as e:
        results["supabase_error"] = str(e)
    return results


# ═══════════════════════════════════════════════════════════
# HISTORY ENDPOINT
# ═══════════════════════════════════════════════════════════
@router.post("/history")
async def get_history(req: HistoryRequest):
    try:
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT question, answer, created_at
                   FROM conversations
                   WHERE user_email = $1
                   ORDER BY created_at DESC
                   LIMIT 50""",
                req.user_email
            )
        rows_list = list(reversed(rows))
        return {
            "success": True,
            "history": [
                {"question": r["question"], "answer": r["answer"], "time": str(r["created_at"] or "")}
                for r in rows_list
            ],
            "count": len(rows_list)
        }
    except Exception as e:
        return {"success": False, "history": [], "count": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════
# ASK — ENDPOINT PRINCIPAL
# ═══════════════════════════════════════════════════════════
@router.post("/ask")
async def ask(req: ChatRequest, request: Request):

    # 1. RATE LIMITING
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    question = (req.question or "").strip()
    if not question and not req.image:
        raise HTTPException(400, "Question vide")

    key = os.environ.get("GROQ_API_KEY", "")

    # 2. IMAGE
    if req.image:
        b64 = req.image
        if "base64," in b64:
            b64 = b64.split("base64,")[1]
        answer = call_groq_vision(question, b64, req.image_type or "image/jpeg")
        if req.user_email and req.user_email != "admin":
            await save_conv(req.user_email, "[image] " + question, answer)
        return {"answer": answer, "sources": [], "mode": "vision"}

    lang   = detect_lang(question)
    q_type = classify_question(question)

    # 3. RECHERCHE DOCUMENTS
    docs    = []
    context = ""
    if q_type in ("question", "urgent", "source"):
        docs = await search_docs(question, limit=5)
        if docs:
            docs = sorted(docs, key=lambda d: d["score"], reverse=True)[:3]
            parts = []
            for d in docs:
                prefix = f"[{d['filename']}]\n" if d['filename'] else ""
                parts.append(prefix + d['content'][:500])
            context = "\n---\n".join(parts)

    # 4. HISTORIQUE
    history = []
    if req.user_email and req.user_email != "admin":
        history = await load_history(req.user_email, limit=4)

    # 5. PROMPT
    if q_type == "greeting":
        system = "Tu es l'assistant SNTF. Réponds chaleureusement en 1-2 phrases et propose ton aide pour les questions techniques ou procédures SNTF."
    elif q_type == "urgent":
        system = f"""Assistant terrain SNTF — MODE URGENCE.
RÈGLES STRICTES :
1. Réponds en MAX 6 lignes
2. Si tu as la solution → étapes numérotées immédiates
3. Si info manquante → UNE seule question ciblée
4. Jamais d'introduction. Jamais de paragraphe.
5. Termine par : "Résolu ?" ou "Si persiste → Centre de contrôle"
6. Langue : {"arabe" if lang == "ar" else "français"}"""
    elif q_type == "source":
        system = "Assistant SNTF. Réponds avec la réponse ET indique la source : [nom du fichier], chunk [numéro]. Sois précis."
    else:
        system = f"""Assistant terrain SNTF — expert ferroviaire algérien.
RÈGLES :
1. Réponse directe et courte — max 8 lignes
2. Procédure → étapes numérotées courtes
3. Pas d'introduction
4. Si info insuffisante → 1 seule question ciblée
5. Langue : {"arabe" if lang == "ar" else "français"}"""

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    user_content = f"[Doc SNTF]: {context}\n\n[Question]: {question}" if context else question
    messages.append({"role": "user", "content": user_content})

    # 6. BOUTONS RÉPONSE RAPIDE
    def get_quick_replies(q_type: str, has_docs: bool) -> list:
        if q_type == "greeting":   return []
        if q_type == "urgent":     return ["Résolu", "Ça persiste", "Centre de contrôle"]
        if has_docs:               return ["Sources du document", "Plus de détails", "Étape suivante"]
        return ["Plus de détails", "Autre question", "Assistance"]

    quick_replies = get_quick_replies(q_type, bool(docs))
    sources = list(set(d["filename"] for d in docs if d["filename"]))

    # 7. STREAMING
    async def generate():
        full_answer = ""

        yield "data: " + json.dumps({
            "sources": sources,
            "web": False,
            "quick_replies": quick_replies
        }) + "\n\n"

        if not key:
            yield "data: " + json.dumps({"token": "Clé GROQ_API_KEY manquante sur le serveur."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        max_attempts = 2
        for attempt in range(max_attempts):
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
                    error_text = r.text[:300] if r.text else f"Erreur HTTP {r.status_code}"
                    print(f"[groq] Erreur {r.status_code}: {error_text}")
                    if r.status_code == 429 and attempt < max_attempts - 1:
                        time.sleep(2)
                        continue
                    yield "data: " + json.dumps({"token": f"Service IA indisponible (erreur {r.status_code}). Réessayez."}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

                for line in r.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8", errors="ignore")
                    if not line_str.startswith("data: "):
                        continue
                    data = line_str[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            full_answer += token
                            yield "data: " + json.dumps({"token": token}) + "\n\n"
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
                break

            except requests.exceptions.Timeout:
                print(f"[groq] Timeout (tentative {attempt + 1}/{max_attempts})")
                if attempt < max_attempts - 1:
                    yield "data: " + json.dumps({"token": "Connexion lente, nouvelle tentative..."}) + "\n\n"
                    full_answer = ""
                    continue
                yield "data: " + json.dumps({"token": "\n\nLe service IA met trop de temps. Réessayez."}) + "\n\n"
            except requests.exceptions.ConnectionError:
                yield "data: " + json.dumps({"token": "Impossible de joindre le service IA."}) + "\n\n"
                break
            except Exception as e:
                print(f"[stream] Exception: {e}")
                yield "data: " + json.dumps({"token": "Une erreur inattendue s'est produite."}) + "\n\n"
                break

        yield "data: [DONE]\n\n"

        if full_answer and len(full_answer) > 10 and req.user_email and req.user_email != "admin":
            await save_conv(req.user_email, question, full_answer)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )
