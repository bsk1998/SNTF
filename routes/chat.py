from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_db
import os
import requests
import numpy as np

router = APIRouter()

# Même modèle que dans n8n : llama-3.3-70b-versatile, température 0.1
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Même modèle embeddings que dans n8n : sentence-transformers/all-MiniLM-L6-v2
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Même table que dans n8n : n8n_vectors
VECTOR_TABLE = "n8n_vectors"

# Même prompt système que dans n8n
SYSTEM_PROMPT = """Tu es l'assistant officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Tu réponds UNIQUEMENT en te basant sur les documents officiels SNTF disponibles.
Tu cites toujours tes sources avec le nom du document et la page.
Si tu ne trouves pas l'information dans les documents, dis-le clairement.
Tu réponds en français. Tu es professionnel, précis et utile."""

class ChatRequest(BaseModel):
    chatInput: str  # Compatible avec l'interface chat n8n
    bot_id: int = 1

class AskRequest(BaseModel):
    question: str
    bot_id: int = 1

def get_embedding(text: str) -> list:
    """
    Même que le nœud 'Embeddings HuggingFace Inference' dans n8n
    Modèle: sentence-transformers/all-MiniLM-L6-v2
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        if response.status_code == 200:
            embedding = response.json()
            if isinstance(embedding[0], list):
                embedding = np.mean(embedding, axis=0).tolist()
            return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
    return None

def search_documents(question: str, limit: int = 5) -> list:
    """
    Equivalent du nœud 'Recherche Documents SNTF' (retrieve-as-tool)
    Table: n8n_vectors
    """
    embedding = get_embedding(question)
    if not embedding:
        return []

    conn = get_db()
    cur = conn.cursor()
    try:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        cur.execute(
            f"""SELECT content, metadata,
               1 - (embedding <=> %s::vector) as similarity
               FROM {VECTOR_TABLE}
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (embedding_str, embedding_str, limit)
        )
        results = cur.fetchall()
        return [
            {"content": r[0], "metadata": r[1], "similarity": float(r[2])}
            for r in results
            if float(r[2]) > 0.3
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def call_groq(question: str, context: list) -> str:
    """
    Equivalent du nœud 'Groq LLM' dans n8n
    Modèle: llama-3.3-70b-versatile, température: 0.1
    """
    if context:
        context_text = "\n\n---\n\n".join([
            f"Source: {c.get('metadata', {})}\nContenu: {c['content']}"
            for c in context
        ])
        full_system = SYSTEM_PROMPT + f"\n\nDOCUMENTS DISPONIBLES:\n{context_text}"
    else:
        full_system = SYSTEM_PROMPT + "\n\nAucun document trouvé pour cette question."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    raise HTTPException(500, f"Erreur Groq: {response.text}")

@router.post("/message")
def message(request: ChatRequest):
    """
    Compatible avec l'URL webhook n8n :
    /webhook/06119fbf-8a1f-4f6b-87ef-6dd70ec3ac80/chat
    Retourne: { output: "..." }
    """
    context = search_documents(request.chatInput)
    answer = call_groq(request.chatInput, context)
    return {"output": answer, "sources_found": len(context)}

@router.post("/ask")
def ask(request: AskRequest):
    """Endpoint alternatif pour l'app Flutter"""
    context = search_documents(request.question)
    answer = call_groq(request.question, context)
    return {
        "success": True,
        "answer": answer,
        "sources_found": len(context)
    }
