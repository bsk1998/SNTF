from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from database import get_db
import os
import requests
import numpy as np
import io

router = APIRouter()

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")

# Même table que dans n8n
VECTOR_TABLE = "n8n_vectors"

# Même paramètres que dans n8n : chunkSize=512, chunkOverlap=50
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Equivalent du nœud 'Extraction Texte PDF' dans n8n"""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(400, f"Impossible de lire le PDF: {str(e)}")

def split_into_chunks(text: str) -> list:
    """
    Equivalent du nœud 'Découpage en Chunks' dans n8n
    chunkSize: 512 tokens, chunkOverlap: 50
    """
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
    """
    Equivalent du nœud 'Embeddings HuggingFace Inference1' dans n8n
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

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_name: str = Form(...),
    category: str = Form(default="General"),
    admin_key: str = Form(...)
):
    """
    Equivalent du flux Upload dans n8n :
    Formulaire → Extraction PDF → Vérifier texte → Préparer →
    Charger pour Découpage → Découpage en Chunks → Stocker dans Supabase →
    Confirmation Upload
    """
    # Vérification clé admin
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")

    # Vérifier que c'est un PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés")

    # Extraction du texte (nœud 'Extraction Texte PDF')
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)

    # Vérification texte extrait (nœud 'Texte Extrait ?')
    if not text:
        raise HTTPException(400, "Impossible d'extraire le texte du PDF. Vérifiez que le PDF n'est pas scanné.")

    # Préparer le document (nœud 'Préparer Document')
    metadata_base = {
        "filename": document_name,
        "category": category,
        "source": file.filename
    }

    # Découpage en chunks (nœud 'Découpage en Chunks')
    chunks = split_into_chunks(text)

    if not chunks:
        raise HTTPException(400, "Le document est trop court ou vide")

    # Stocker dans Supabase (nœud 'Stocker dans Supabase')
    conn = get_db()
    cur = conn.cursor()
    stored = 0

    try:
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if not embedding:
                continue

            metadata = {
                **metadata_base,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }

            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            cur.execute(
                f"""INSERT INTO {VECTOR_TABLE} (content, metadata, embedding)
                   VALUES (%s, %s::jsonb, %s::vector)""",
                (chunk, str(metadata).replace("'", '"'), embedding_str)
            )
            stored += 1

        conn.commit()

        # Confirmation Upload (nœud 'Confirmation Upload')
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
    """Liste tous les documents indexés dans n8n_vectors"""
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
    """Supprimer un document de la base vectorielle"""
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
        return {
            "success": True,
            "message": f"Document '{filename}' supprimé",
            "chunks_deleted": deleted
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()
