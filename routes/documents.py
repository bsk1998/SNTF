from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pypdf import PdfReader
import io
import os
import json
import hashlib
from database import get_db

router = APIRouter()

ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")
VECTOR_TABLE = "n8n_vectors"

def get_embedding(text: str) -> list:
    """
    Embedding simple sans API externe — utilise sentence-transformers local
    ou fallback vers un vecteur basé sur le hash du texte
    """
    # Méthode 1 : sentence-transformers local (si installé)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text).tolist()
        print(f"✅ Embedding local OK: {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        print(f"sentence-transformers failed: {e}")

    # Méthode 2 : HuggingFace API (si clé dispo)
    HF_API_KEY = os.environ.get("HF_API_KEY")
    if HF_API_KEY:
        try:
            import requests
            import numpy as np
            response = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": text[:500], "options": {"wait_for_model": True}},
                timeout=60
            )
            print(f"📡 HuggingFace status: {response.status_code}")
            if response.status_code == 200:
                embedding = response.json()
                if isinstance(embedding[0], list):
                    import numpy as np
                    embedding = np.mean(embedding, axis=0).tolist()
                print(f"✅ HuggingFace OK: {len(embedding)} dimensions")
                return embedding
            else:
                print(f"❌ HuggingFace erreur: {response.text[:200]}")
        except Exception as e:
            print(f"HuggingFace API failed: {e}")

    # Méthode 3 : Embedding de secours basé sur TF-IDF simple (384 dimensions)
    try:
        import hashlib
        import math
        words = text.lower().split()
        vector = [0.0] * 384
        for i, word in enumerate(words[:384]):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % 384
            vector[idx] += 1.0 / (i + 1)
        # Normaliser
        norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
        vector = [x / norm for x in vector]
        print(f"✅ Embedding de secours OK: {len(vector)} dimensions")
        return vector
    except Exception as e:
        print(f"Fallback embedding failed: {e}")

    return None

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
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
            raise HTTPException(400, "Impossible d'extraire le texte.")

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
                print(f"⚠️ Chunk {i} — embedding échoué")
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
                f"""INSERT INTO {VECTOR_TABLE} (text, metadata, embedding)
                   VALUES (%s, %s::jsonb, %s::vector)""",
                (chunk, metadata, embedding_str)
            )
            stored += 1
            print(f"💾 Chunk {i+1}/{len(chunks)} sauvegardé")

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ TOTAL: {stored} chunks sauvegardés dans Supabase")

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
        print(f"❌ Erreur: {e}")
        raise HTTPException(500, f"Erreur: {str(e)}")

@router.get("/list")
def list_documents(admin_key: str = ""):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT metadata->>'filename', metadata->>'category', COUNT(*)
            FROM {VECTOR_TABLE}
            WHERE metadata->>'filename' IS NOT NULL
            GROUP BY metadata->>'filename', metadata->>'category'
            ORDER BY metadata->>'filename'
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
