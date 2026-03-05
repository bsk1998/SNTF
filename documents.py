from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from database import get_db
import os
import requests
import numpy as np
import io
import json

router = APIRouter()

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")
VECTOR_TABLE = "n8n_vectors"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Essaie plusieurs librairies PDF pour extraire le texte"""
    text = ""

    # Méthode 1 : pypdf (nouveau nom de PyPDF2)
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if text.strip():
            print(f"✅ pypdf: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"pypdf failed: {e}")

    # Méthode 2 : PyPDF2 (ancien nom)
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if text.strip():
            print(f"✅ PyPDF2: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"PyPDF2 failed: {e}")

    # Méthode 3 : pdfminer
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(io.BytesIO(file_bytes))
        if text and text.strip():
            print(f"✅ pdfminer: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"pdfminer failed: {e}")

    # Méthode 4 : pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if text.strip():
            print(f"✅ pdfplumber: {len(text)} caractères extraits")
            return text.strip()
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    raise HTTPException(400, "Impossible d'extraire le texte du PDF. Vérifiez que le PDF contient du texte sélectionnable.")

def split_into_chunks(text: str) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk.strip()) >= 50:
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

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

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_name: str = Form(...),
    category: str = Form(default="General"),
    admin_key: str = Form(...)
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés")

    file_bytes = await file.read()
    print(f"📄 Fichier reçu: {file.filename} ({len(file_bytes)} bytes)")

    text = extract_text_from_pdf(file_bytes)
    print(f"📝 Texte extrait: {len(text)} caractères")

    if not text:
        raise HTTPException(400, "Impossible d'extraire le texte du PDF.")

    metadata_base = {
        "filename": document_name,
        "category": category,
        "source": file.filename
    }

    chunks = split_into_chunks(text)
    print(f"🔪 Chunks créés: {len(chunks)}")

    if not chunks:
        raise HTTPException(400, "Le document est trop court ou vide")

    conn = get_db()
    cur = conn.cursor()
    stored = 0

    try:
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if not embedding:
                continue
            metadata = {**metadata_base, "chunk_index": i, "total_chunks": len(chunks)}
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            cur.execute(
                f"""INSERT INTO {VECTOR_TABLE} (content, metadata, embedding)
                   VALUES (%s, %s::jsonb, %s::vector)""",
                (chunk, json.dumps(metadata), embedding_str)
            )
            stored += 1

        conn.commit()
        print(f"✅ {stored} chunks sauvegardés dans Supabase")

        return {
            "success": True,
            "message": f"✅ Document '{document_name}' indexé avec succès !",
            "filename": document_name,
            "category": category,
            "chunks_stored": stored,
            "total_chunks": len(chunks)
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(500, f"Erreur lors du stockage: {str(e)}")
    finally:
        cur.close()
        conn.close()

@router.get("/list")
def list_documents(admin_key: str):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""SELECT
               metadata->>'filename' as filename,
               metadata->>'category' as category,
               COUNT(*) as chunks
               FROM {VECTOR_TABLE}
               WHERE metadata->>'filename' IS NOT NULL
               GROUP BY metadata->>'filename', metadata->>'category'
               ORDER BY filename"""
        )
        docs = cur.fetchall()
        return {
            "success": True,
            "total_documents": len(docs),
            "documents": [
                {"filename": d[0], "category": d[1], "chunks": int(d[2])}
                for d in docs
            ]
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()

@router.delete("/delete")
def delete_document(filename: str, admin_key: str):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            f"DELETE FROM {VECTOR_TABLE} WHERE metadata->>'filename' = %s",
            (filename,)
        )
        deleted = cur.rowcount
        conn.commit()
        return {"success": True, "message": f"Document '{filename}' supprimé", "chunks_deleted": deleted}
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()
