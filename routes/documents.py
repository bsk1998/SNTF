from fastapi import APIRouter, UploadFile, File, HTTPException
from pypdf import PdfReader
import io
import os
import requests
import numpy as np
from database import get_db

router = APIRouter()

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_TABLE = "n8n_vectors"

def get_embedding(text: str) -> list:
    """Génère l'embedding avec HuggingFace"""
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
    """Extrait le texte d'un PDF"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Erreur lecture PDF: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    """Découpe le texte en chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload et traite un PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés")
    
    try:
        # Lire le PDF
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text:
            raise HTTPException(400, "Impossible d'extraire le texte du PDF")
        
        # Découper en chunks
        chunks = split_text_into_chunks(text)
        
        # Stocker dans Supabase
        conn = get_db()
        cur = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if not embedding:
                continue
            
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            metadata = {"source": file.filename, "chunk": i+1}
            
            cur.execute(
                f"""INSERT INTO {VECTOR_TABLE} (content, metadata, embedding)
                   VALUES (%s, %s, %s::vector)""",
                (chunk, str(metadata), embedding_str)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Erreur: {str(e)}")

@router.get("/list")
def list_documents():
    """Liste tous les documents"""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(DISTINCT metadata->>'source') FROM {VECTOR_TABLE}")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"total_documents": count}
    except Exception as e:
        return {"error": str(e)}
        
