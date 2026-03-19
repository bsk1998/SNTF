from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os, requests, json, hashlib, math

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    image: Optional[str] = None
    image_type: Optional[str] = None
    images: Optional[list] = None
    pdfs: Optional[list] = None
    memory: Optional[object] = None  # accepte string, list ou null
    user_email: Optional[str] = None
    stream: Optional[bool] = True

class HistoryRequest(BaseModel):
    user_email: str

# ═══════════════════════════════════════════
# DIAGNOSTIC COMPLET
# ═══════════════════════════════════════════
@router.get("/test")
def test_all():
    results = {}

    # 1. Clé Groq
    key = os.environ.get("GROQ_API_KEY", "")
    results["groq_key_present"] = bool(key)
    results["groq_key_prefix"] = key[:10] + "..." if key else "ABSENTE"

    # 2. Test Groq API
    if key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": "dis juste: OK"}],
                      "max_tokens": 5},
                timeout=15
            )
            results["groq_status"] = r.status_code
            if r.status_code == 200:
                results["groq_response"] = r.json()["choices"][0]["message"]["content"]
            else:
                results["groq_error"] = r.text[:300]
        except Exception as e:
            results["groq_exception"] = str(e)

    # 3. Test Supabase connexion
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        results["supabase"] = "connecté"
        cur.close()
        conn.close()
    except Exception as e:
        results["supabase_error"] = str(e)
        results["supabase"] = "ERREUR"

    # 4. Test table document_chunks
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        count = cur.fetchone()[0]
        results["document_chunks_count"] = count
        cur.close()
        conn.close()
    except Exception as e:
        results["document_chunks_error"] = str(e)

    # 5. Test table conversations
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('public.conversations')")
        exists = cur.fetchone()[0]
        results["conversations_table"] = "existe" if exists else "ABSENTE"
        cur.close()
        conn.close()
    except Exception as e:
        results["conversations_error"] = str(e)

    # 6. Test streaming Groq
    if key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": "dis juste: OK"}],
                      "max_tokens": 5,
                      "stream": True},
                stream=True,
                timeout=15
            )
            results["groq_stream_status"] = r.status_code
            tokens = []
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            t = chunk["choices"][0]["delta"].get("content", "")
                            if t:
                                tokens.append(t)
                        except:
                            pass
            results["groq_stream_response"] = "".join(tokens)
            results["groq_stream"] = "OK" if tokens else "VIDE"
        except Exception as e:
            results["groq_stream_exception"] = str(e)
            results["groq_stream"] = "ERREUR"

    return results


# ═══════════════════════════════════════════
# HISTORY
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
        cur.close()
        conn.close()
        rows.reverse()
        history = [{"question": r[0], "answer": r[1], "time": str(r[2]) if r[2] else ""} for r in rows]
        return {"success": True, "history": history, "count": len(history)}
    except Exception as e:
        return {"success": False, "history": [], "count": 0, "error": str(e)}


# ═══════════════════════════════════════════
# ASK — STREAMING
# ═══════════════════════════════════════════
@router.post("/ask")
async def ask(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "Question vide")

    key = os.environ.get("GROQ_API_KEY", "")

    # Contexte documents — ne bloque JAMAIS même si erreur
    context = ""
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
            "SELECT content, metadata FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT 5",
            (emb_str,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if rows:
            parts = []
            for r in rows:
                if r[0]:
                    try:
                        meta = json.loads(r[1]) if r[1] else {}
                    except:
                        meta = {}
                    fname = meta.get("filename", "")
                    prefix = f"[{fname}]\n" if fname else ""
                    parts.append(prefix + r[0])
            context = "\n\n---\n\n".join(parts)
    except Exception as e:
        print(f"[context error - non bloquant] {e}")
        context = ""

    # Historique — ne bloque JAMAIS
    history = []
    try:
        if req.user_email and req.user_email != "admin":
            from database import get_db
            conn = get_db()
            cur = conn.cursor()
            # Créer la table si elle n'existe pas
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
                "SELECT question, answer FROM conversations WHERE user_email=%s ORDER BY created_at DESC LIMIT 4",
                (req.user_email,)
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            for r in reversed(rows):
                history.append({"role": "user", "content": r[0]})
                history.append({"role": "assistant", "content": r[1][:400]})
    except Exception as e:
        print(f"[history error - non bloquant] {e}")
        history = []

    # Messages Groq
    system = """Tu es l'assistant officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT en te basant sur les documents fournis dans le contexte
2. Recopie les informations EXACTEMENT comme elles apparaissent dans les documents — ne résume pas, ne reformule pas, ne complète pas
3. Si le contexte contient la réponse, cite-la intégralement et clairement
4. Si le contexte ne contient PAS la réponse, dis simplement : "Je n'ai pas cette information dans les documents disponibles."
5. Ne jamais inventer, déduire ou compléter avec des informations hors des documents
6. Si la question est en arabe, réponds en arabe
7. Réponds de façon structurée et professionnelle"""

    messages = [{"role": "system", "content": system}]
    messages.extend(history)

    if context:
        messages.append({"role": "user", "content": f"""DOCUMENTS SNTF (source officielle) :
{context}

QUESTION : {question}

Réponds en te basant STRICTEMENT sur les documents ci-dessus. Cite les informations exactes."""})
    else:
        messages.append({"role": "user", "content": f"{question}\n\n(Aucun document disponible — réponds de façon générale sur la SNTF)"})

    # Streaming SSE
    def generate():
        full_answer = ""

        # Premier chunk : sources
        yield "data: " + json.dumps({"sources": [], "web": False}) + "\n\n"

        if not key:
            yield "data: " + json.dumps({"token": "⚠️ GROQ_API_KEY manquante sur Render. Allez dans Environment et ajoutez la clé."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "stream": True
                },
                stream=True,
                timeout=45
            )

            if r.status_code != 200:
                err = f"⚠️ Erreur Groq {r.status_code}: {r.text[:200]}"
                print(f"[groq error] {err}")
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
            yield "data: " + json.dumps({"token": "⚠️ Timeout — Groq met trop de temps. Réessayez."}) + "\n\n"
        except Exception as e:
            print(f"[stream error] {e}")
            yield "data: " + json.dumps({"token": f"⚠️ Erreur: {str(e)}"}) + "\n\n"

        yield "data: [DONE]\n\n"

        # Sauvegarder la conversation
        if full_answer and req.user_email and req.user_email != "admin":
            try:
                from database import get_db
                conn = get_db()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO conversations (user_email, question, answer) VALUES (%s, %s, %s)",
                    (req.user_email, question[:1000], full_answer[:5000])
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[save error - non bloquant] {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
