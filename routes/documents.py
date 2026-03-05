from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pypdf import PdfReader
import io
import os
import requests
import numpy as np
import json
from database import get_db

router = APIRouter()

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")
VECTOR_TABLE = "n8n_vectors"

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

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    # Méthode 1 : pypdf
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if text.strip():
            print(f"✅ pypdf: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"pypdf failed: {e}")

    # Méthode 2 : pdfminer
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(io.BytesIO(pdf_bytes))
        if text and text.strip():
            print(f"✅ pdfminer: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"pdfminer failed: {e}")

    # Méthode 3 : pdfplumber
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if text.strip():
            print(f"✅ pdfplumber: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    return ""

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) >= 50:
            chunks.append(chunk)
    return chunks

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    document_name: str = Form(default=""),
    category: str = Form(default="General"),
    admin_key: str = Form(default="")
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés")

    try:
        pdf_bytes = await file.read()
        print(f"📄 Fichier reçu: {file.filename} ({len(pdf_bytes)} bytes)")

        text = extract_text_from_pdf(pdf_bytes)
        print(f"📝 Texte extrait: {len(text)} caractères")

        if not text:
            raise HTTPException(400, "Impossible d'extraire le texte. PDF scanné ou protégé ?")

        chunks = split_text_into_chunks(text)
        print(f"🔪 Chunks créés: {len(chunks)}")

        if not chunks:
            raise HTTPException(400, "Document trop court ou vide")

        doc_name = document_name if document_name else file.filename.replace('.pdf', '')

        conn = get_db()
        cur = conn.cursor()
        stored = 0

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if not embedding:
                continue

            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            metadata = json.dumps({
                "source": file.filename,
                "filename": doc_name,
                "category": category,
                "chunk": i + 1,
                "total_chunks": len(chunks)
            })

            cur.execute(
                f"""INSERT INTO {VECTOR_TABLE} (content, metadata, embedding)
                   VALUES (%s, %s::jsonb, %s::vector)""",
                (chunk, metadata, embedding_str)
            )
            stored += 1

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ {stored} chunks sauvegardés")

        return {
            "success": True,
            "filename": doc_name,
            "category": category,
            "chunks_stored": stored,
            "total_chunks": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur: {str(e)}")

@router.get("/list")
def list_documents(admin_key: str = ""):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT metadata->>'filename' as filename,
                   metadata->>'category' as category,
                   COUNT(*) as chunks
            FROM {VECTOR_TABLE}
            WHERE metadata->>'filename' IS NOT NULL
            GROUP BY metadata->>'filename', metadata->>'category'
            ORDER BY filename
        """)
        docs = cur.fetchall()
        cur.close()
        conn.close()
        return {
            "success": True,
            "total_documents": len(docs),
            "documents": [{"filename": d[0], "category": d[1], "chunks": int(d[2])} for d in docs]
        }
    except Exception as e:
        return {"error": str(e)}

@router.delete("/delete")
def delete_document(filename: str, admin_key: str = ""):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {VECTOR_TABLE} WHERE metadata->>'filename' = %s", (filename,))
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "chunks_deleted": deleted}
    except Exception as e:
        raise HTTPException(500, str(e))
