from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_db
import os
import requests
import numpy as np

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_TABLE = "n8n_vectors"

# Seuil : si similarité < 0.5 ou moins de 2 résultats → chercher internet
PDF_SIMILARITY_THRESHOLD = 0.5
PDF_MIN_RESULTS = 2

SYSTEM_PROMPT_PDF = """Tu es l'assistant officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Tu réponds en te basant sur les documents officiels SNTF disponibles.
Tu cites toujours tes sources avec le nom du document.
Tu réponds en français. Tu es professionnel, précis et utile."""

SYSTEM_PROMPT_WEB = """Tu es l'assistant officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Les documents internes SNTF ne contiennent pas d'information suffisante sur ce sujet.
Tu vas compléter avec des informations générales fiables.
Précise toujours que l'information provient de sources externes et non des documents officiels SNTF.
Tu réponds en français. Tu es professionnel, précis et utile."""

SYSTEM_PROMPT_BOTH = """Tu es l'assistant officiel de la SNTF (Société Nationale des Transports Ferroviaires d'Algérie).
Tu as accès aux documents officiels SNTF ET à des informations complémentaires d'internet.
Privilégie toujours les documents officiels SNTF. Utilise internet uniquement pour compléter.
Indique clairement quelle source tu utilises pour chaque partie de ta réponse.
Tu réponds en français. Tu es professionnel, précis et utile."""

class ChatRequest(BaseModel):
    chatInput: str
    bot_id: int = 1

class AskRequest(BaseModel):
    question: str
    bot_id: int = 1

def get_embedding(text: str) -> list:
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
            for r in results if float(r[2]) > 0.3
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def pdf_results_sufficient(pdf_results: list) -> bool:
    if len(pdf_results) < PDF_MIN_RESULTS:
        return False
    top_similarity = max(r["similarity"] for r in pdf_results)
    return top_similarity >= PDF_SIMILARITY_THRESHOLD

def search_web(question: str) -> str:
    """Recherche web via Groq avec tool calling"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 800,
        "messages": [
            {
                "role": "system",
                "content": "Recherche des informations fiables sur internet et résume-les en français de façon factuelle."
            },
            {
                "role": "user",
                "content": f"Trouve des informations sur: {question}"
            }
        ]
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"].get("content", "")
    except Exception as e:
        print(f"Web search error: {e}")
    return ""

def call_groq(question: str, context: list, web_content: str = "") -> str:
    has_pdf = len(context) > 0
    has_web = bool(web_content)

    if has_pdf and has_web:
        pdf_text = "\n\n---\n\n".join([f"📄 {c.get('metadata',{})}\n{c['content']}" for c in context])
        full_system = SYSTEM_PROMPT_BOTH + f"\n\n📚 DOCUMENTS OFFICIELS SNTF:\n{pdf_text}\n\n🌐 INFORMATIONS INTERNET:\n{web_content}"
    elif has_pdf:
        pdf_text = "\n\n---\n\n".join([f"📄 {c.get('metadata',{})}\n{c['content']}" for c in context])
        full_system = SYSTEM_PROMPT_PDF + f"\n\nDOCUMENTS:\n{pdf_text}"
    elif has_web:
        full_system = SYSTEM_PROMPT_WEB + f"\n\n🌐 INFORMATIONS:\n{web_content}"
    else:
        full_system = SYSTEM_PROMPT_PDF + "\n\nAucune information trouvée. Indique-le clairement à l'utilisateur."

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
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
        headers=headers, json=payload, timeout=30
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    raise HTTPException(500, f"Erreur Groq: {response.text}")

def smart_answer(question: str) -> dict:
    """
    Logique intelligente :
    1. Cherche dans les PDF
    2. Si PDF suffisant (similarité >= 0.5 et >= 2 résultats) → répond avec PDF
    3. Sinon → cherche sur internet → répond avec internet (+ PDF si dispo)
    """
    pdf_results = search_documents(question)

    if pdf_results_sufficient(pdf_results):
        answer = call_groq(question, pdf_results)
        return {"answer": answer, "sources_found": len(pdf_results), "used_web": False, "source_type": "pdf"}
    else:
        print(f"PDF insuffisant ({len(pdf_results)} résultats) → recherche web")
        web_content = search_web(question)
        answer = call_groq(question, pdf_results, web_content)
        source_type = "both" if (pdf_results and web_content) else ("web" if web_content else "none")
        return {"answer": answer, "sources_found": len(pdf_results), "used_web": True, "source_type": source_type}

@router.post("/message")
def message(request: ChatRequest):
    result = smart_answer(request.chatInput)
    return {"output": result["answer"], "sources_found": result["sources_found"]}

@router.post("/ask")
def ask(request: AskRequest):
    result = smart_answer(request.question)
    return {
        "success": True,
        "answer": result["answer"],
        "sources_found": result["sources_found"],
        "used_web": result["used_web"],
        "source_type": result["source_type"]
    }
