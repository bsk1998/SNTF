from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math, re, time
from collections import defaultdict
from threading import Lock

router = APIRouter()

# ═══════════════════════════════════════════════════════════
# RATE LIMITING — sans dépendance externe
# Simple compteur en mémoire par IP
# Limite : 30 requêtes / minute / IP sur /ask
# ═══════════════════════════════════════════════════════════
_rate_store: dict = defaultdict(list)
_rate_lock = Lock()

RATE_LIMIT_REQUESTS = 30   # max requêtes
RATE_LIMIT_WINDOW   = 60   # sur 60 secondes

def check_rate_limit(ip: str) -> None:
    """
    Lève HTTPException 429 si l'IP dépasse la limite.
    Nettoyage automatique des anciennes entrées.
    """
    now = time.time()
    with _rate_lock:
        # Garder seulement les timestamps dans la fenêtre
        _rate_store[ip] = [
            t for t in _rate_store[ip]
            if now - t < RATE_LIMIT_WINDOW
        ]
        if len(_rate_store[ip]) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Trop de requêtes. Limite : {RATE_LIMIT_REQUESTS} messages par minute. Réessayez dans {RATE_LIMIT_WINDOW} secondes."
            )
        _rate_store[ip].append(now)

def get_client_ip(request: Request) -> str:
    """Récupère l'IP réelle même derrière un proxy Render/Cloudflare."""
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
# DIAGNOSTIC
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════
_hf_model = None

def get_embedding(text: str) -> list:
    global _hf_model
    hf_key = os.environ.get("HF_API_KEY", "")

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

    # Fallback hash
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:200]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x / norm for x in vector]

# ═══════════════════════════════════════════════════════════
# RECHERCHE DOCUMENTS
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# MÉMOIRE CONVERSATION
# ═══════════════════════════════════════════════════════════
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
# HISTORY ENDPOINT
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# ASK — ENDPOINT PRINCIPAL avec rate limiting + streaming robuste
# ═══════════════════════════════════════════════════════════
@router.post("/ask")
async def ask(req: ChatRequest, request: Request):

    # ── 1. RATE LIMITING ──────────────────────────────────
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    question = (req.question or "").strip()
    if not question and not req.image:
        raise HTTPException(400, "Question vide")

    key = os.environ.get("GROQ_API_KEY", "")

    # ── 2. IMAGE ──────────────────────────────────────────
    if req.image:
        b64 = req.image
        if "base64," in b64:
            b64 = b64.split("base64,")[1]
        answer = call_groq_vision(question, b64, req.image_type or "image/jpeg")
        if req.user_email and req.user_email != "admin":
            save_conv(req.user_email, "[image] " + question, answer)
        return {"answer": answer, "sources": [], "mode": "vision"}

    lang   = detect_lang(question)
    q_type = classify_question(question)

    # ── 3. RECHERCHE DOCUMENTS ────────────────────────────
    docs    = []
    context = ""
    if q_type in ("question", "urgent", "source"):
        docs = search_docs(question, limit=5)
        if docs:
            docs = sorted(docs, key=lambda d: d["score"], reverse=True)[:3]
            parts = []
            for d in docs:
                prefix = f"[{d['filename']}]\n" if d['filename'] else ""
                parts.append(prefix + d['content'][:500])
            context = "\n---\n".join(parts)

    # ── 4. HISTORIQUE ─────────────────────────────────────
    history = []
    if req.user_email and req.user_email != "admin":
        history = load_history(req.user_email, limit=4)

    # ── 5. PROMPT ─────────────────────────────────────────
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
        system = """Assistant SNTF. Réponds avec la réponse ET indique la source : [nom du fichier], chunk [numéro]. Sois précis."""
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

    # ── 6. BOUTONS RÉPONSE RAPIDE ─────────────────────────
    def get_quick_replies(q_type: str, has_docs: bool) -> list:
        if q_type == "greeting":
            return []
        if q_type == "urgent":
            return ["Résolu", "Ça persiste", "Centre de contrôle"]
        if has_docs:
            return ["Sources du document", "Plus de détails", "Étape suivante"]
        return ["Plus de détails", "Autre question", "Assistance"]

    quick_replies = get_quick_replies(q_type, bool(docs))
    sources = list(set(d["filename"] for d in docs if d["filename"]))

    # ── 7. STREAMING ROBUSTE ──────────────────────────────
    # Corrections vs version originale :
    # - Timeout de 45s avec gestion explicite
    # - Retry automatique 1 fois si timeout
    # - Message d'erreur clair au client via SSE
    # - Nettoyage du buffer même en cas d'exception

    def generate():
        full_answer = ""

        # Métadonnées en premier chunk
        yield "data: " + json.dumps({
            "sources": sources,
            "web": False,
            "quick_replies": quick_replies
        }) + "\n\n"

        if not key:
            yield "data: " + json.dumps({"token": "Clé GROQ_API_KEY manquante sur le serveur."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        # Tentatives : 1 essai normal + 1 retry si timeout
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
                    timeout=45  # timeout strict
                )

                # Erreur HTTP de Groq
                if r.status_code != 200:
                    error_text = r.text[:300] if r.text else f"Erreur HTTP {r.status_code}"
                    print(f"[groq] Erreur {r.status_code}: {error_text}")

                    if r.status_code == 429 and attempt < max_attempts - 1:
                        # Quota Groq dépassé — retry après 2s
                        import time as t
                        t.sleep(2)
                        continue

                    yield "data: " + json.dumps({
                        "token": f"Le service IA est temporairement indisponible (erreur {r.status_code}). Réessayez dans quelques instants."
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Lecture du stream token par token
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

                # Succès — on sort de la boucle de retry
                break

            except requests.exceptions.Timeout:
                print(f"[groq] Timeout (tentative {attempt + 1}/{max_attempts})")
                if attempt < max_attempts - 1:
                    # Retry silencieux
                    yield "data: " + json.dumps({"token": "Connexion lente, nouvelle tentative..."}) + "\n\n"
                    full_answer = ""
                    continue
                # Dernier essai échoué
                yield "data: " + json.dumps({
                    "token": "\n\nLe service IA met trop de temps à répondre. Réessayez votre question."
                }) + "\n\n"

            except requests.exceptions.ConnectionError:
                yield "data: " + json.dumps({
                    "token": "Impossible de joindre le service IA. Vérifiez votre connexion."
                }) + "\n\n"
                break

            except Exception as e:
                print(f"[stream] Exception inattendue: {e}")
                yield "data: " + json.dumps({
                    "token": "Une erreur inattendue s'est produite. Réessayez."
                }) + "\n\n"
                break

        # Fin du stream
        yield "data: [DONE]\n\n"

        # Sauvegarde uniquement si on a une vraie réponse
        if full_answer and len(full_answer) > 10 and req.user_email and req.user_email != "admin":
            save_conv(req.user_email, question, full_answer)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )
