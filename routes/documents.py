from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import io, os, json, hashlib, math, requests, base64
from database import get_db

router = APIRouter()

ADMIN_KEY   = os.environ.get("ADMIN_KEY", "sntf_admin_2024")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VECTOR_TABLE = "document_chunks"

# ═══════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════
_model = None
def get_embedding(text: str) -> list:
    global _model
    # Méthode 1 : sentence-transformers local
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        return _model.encode(text[:500]).tolist()
    except: pass

    # Méthode 2 : HuggingFace API
    HF_KEY = os.environ.get("HF_API_KEY")
    if HF_KEY:
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_KEY}"},
                json={"inputs": text[:500], "options": {"wait_for_model": True}}, timeout=60
            )
            if r.status_code == 200:
                emb = r.json()
                if isinstance(emb[0], list):
                    import numpy as np
                    emb = np.mean(emb, axis=0).tolist()
                return emb
        except: pass

    # Méthode 3 : Fallback hash
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:384]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x/norm for x in vector]

# ═══════════════════════════════════════
# EXTRACTION TEXTE PDF
# ═══════════════════════════════════════
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        print(f"pypdf failed: {e}")
        return ""

# ═══════════════════════════════════════
# EXTRACTION TEXTE IMAGE via Llama 4 Scout
# ═══════════════════════════════════════
def extract_text_from_image(image_bytes: bytes, mime_type: str, filename: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY manquante pour analyser les images")
    try:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        print(f"🖼️ Analyse image via Llama 4 Scout: {filename} ({len(image_bytes)} bytes)")

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": """Analyse cette image en détail pour créer une base de connaissances.
Extrais et décris :
1. Tout le texte visible (tableaux, titres, données, labels)
2. Le contenu principal de l'image (schéma, photo, graphique, document)
3. Les informations techniques ou chiffres importants
4. Le contexte général de l'image

Réponds en français, de façon complète et structurée. Commence par : 'Cette image montre...'"""},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                ]}],
                "max_tokens": 2048
            },
            timeout=60
        )
        if r.status_code == 200:
            description = r.json()["choices"][0]["message"]["content"]
            print(f"✅ Image analysée: {len(description)} caractères")
            return f"[Image: {filename}]\n\n{description}"
        else:
            print(f"❌ Groq vision erreur: {r.status_code} - {r.text[:200]}")
            raise HTTPException(500, f"Erreur analyse image: {r.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image extraction failed: {e}")
        raise HTTPException(500, f"Erreur analyse image: {str(e)}")

# ═══════════════════════════════════════
# CHUNKS
# ═══════════════════════════════════════
def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """
    Découpage intelligent par phrases avec chevauchement.
    - chunk_size : nombre de caractères max par chunk
    - overlap    : chevauchement entre chunks pour ne pas couper les idées
    """
    # Nettoyer le texte
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)  # max 2 sauts de ligne
    text = re.sub(r' {2,}', ' ', text)          # espaces multiples

    # Découper par paragraphes d'abord
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # Si le paragraphe seul dépasse chunk_size, le découper par phrases
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current and len(current.strip()) >= 50:
                        chunks.append(current.strip())
                    # Chevauchement : garder les derniers mots du chunk précédent
                    words = current.split()
                    overlap_text = " ".join(words[-overlap//5:]) if words else ""
                    current = (overlap_text + " " + sent).strip() if overlap_text else sent
        else:
            if len(current) + len(para) + 2 <= chunk_size:
                current += ("\n\n" if current else "") + para
            else:
                if current and len(current.strip()) >= 50:
                    chunks.append(current.strip())
                words = current.split()
                overlap_text = " ".join(words[-overlap//5:]) if words else ""
                current = (overlap_text + "\n\n" + para).strip() if overlap_text else para

    if current and len(current.strip()) >= 50:
        chunks.append(current.strip())

    return chunks

# ═══════════════════════════════════════
# SAUVEGARDER CHUNKS EN BASE
# ═══════════════════════════════════════
def save_chunks(chunks: list, doc_name: str, category: str, source_filename: str, file_type: str) -> int:
    conn = get_db()
    cur  = conn.cursor()
    stored = 0
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if not embedding:
            continue
        emb_str = "[" + ",".join(map(str, embedding)) + "]"
        metadata = json.dumps({
            "source":       source_filename,
            "filename":     doc_name,
            "category":     category,
            "file_type":    file_type,
            "chunk":        i + 1,
            "total_chunks": len(chunks)
        })
        cur.execute(
            f"INSERT INTO {VECTOR_TABLE} (content, metadata, embedding) VALUES (%s, %s::jsonb, %s::vector)",
            (chunk, metadata, emb_str)
        )
        stored += 1
        print(f"💾 Chunk {i+1}/{len(chunks)} sauvegardé")
    conn.commit()
    cur.close()
    conn.close()
    return stored

# ═══════════════════════════════════════
# ENDPOINT UPLOAD — PDF + IMAGES
# ═══════════════════════════════════════
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    document_name: str = Form(default=""),
    category: str = Form(default="General"),
    admin_key: str = Form(default="")
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")

    filename  = file.filename.lower()
    mime_type = file.content_type or ""

    # Types acceptés
    is_pdf   = filename.endswith('.pdf')
    is_image = filename.endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')) or mime_type.startswith('image/')

    if not is_pdf and not is_image:
        raise HTTPException(400, "Types acceptés : PDF, JPG, PNG, WEBP")

    try:
        file_bytes = await file.read()
        print(f"📁 Fichier reçu: {file.filename} ({len(file_bytes)} bytes) — type: {'PDF' if is_pdf else 'IMAGE'}")

        # Extraire le texte selon le type
        if is_pdf:
            text = extract_text_from_pdf(file_bytes)
            file_type = "pdf"
            if not text:
                raise HTTPException(400, "Impossible d'extraire le texte du PDF")
        else:
            # Détecter le bon mime type
            if filename.endswith('.png'):
                detected_mime = "image/png"
            elif filename.endswith('.webp'):
                detected_mime = "image/webp"
            else:
                detected_mime = "image/jpeg"
            text = extract_text_from_image(file_bytes, detected_mime, file.filename)
            file_type = "image"

        print(f"📝 Texte extrait: {len(text)} caractères")

        chunks = split_text_into_chunks(text)
        print(f"🔪 Chunks créés: {len(chunks)}")

        if not chunks:
            raise HTTPException(400, "Document trop court ou vide")

        doc_name = document_name if document_name else file.filename.rsplit('.', 1)[0]
        stored = save_chunks(chunks, doc_name, category, file.filename, file_type)

        print(f"✅ TOTAL: {stored} chunks sauvegardés")
        return {
            "success":       True,
            "filename":      doc_name,
            "category":      category,
            "file_type":     file_type,
            "chunks_stored": stored,
            "total_chunks":  len(chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise HTTPException(500, f"Erreur: {str(e)}")

# ═══════════════════════════════════════
# LIST
# ═══════════════════════════════════════
@router.get("/list")
def list_documents(admin_key: str = ""):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(f"""
            SELECT metadata->>'filename', metadata->>'category',
                   metadata->>'file_type', COUNT(*)
            FROM {VECTOR_TABLE}
            WHERE metadata->>'filename' IS NOT NULL
            GROUP BY metadata->>'filename', metadata->>'category', metadata->>'file_type'
            ORDER BY metadata->>'filename'
        """)
        docs = cur.fetchall()
        cur.close(); conn.close()
        return {
            "success":         True,
            "total_documents": len(docs),
            "documents": [{"filename": d[0], "category": d[1],
                           "file_type": d[2] or "pdf", "chunks": int(d[3])} for d in docs]
        }
    except Exception as e:
        return {"error": str(e)}

# ═══════════════════════════════════════
# DELETE
# ═══════════════════════════════════════
class DeleteRequest(BaseModel):
    filename: str
    admin_key: str = ""

@router.post("/delete")
def delete_document(req: DeleteRequest):
    if req.admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(f"DELETE FROM {VECTOR_TABLE} WHERE metadata->>'filename' = %s", (req.filename,))
        deleted = cur.rowcount
        conn.commit()
        cur.close(); conn.close()
        return {"success": True, "chunks_deleted": deleted, "filename": req.filename}
    except Exception as e:
        raise HTTPException(500, str(e))




# ═══════════════════════════════════════
# UPLOAD BATCH — Plusieurs fichiers en une requête
# ═══════════════════════════════════════
from typing import List

@router.post("/upload-batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    category: str = Form(default="General"),
    admin_key: str = Form(default="")
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte")

    if not files:
        raise HTTPException(400, "Aucun fichier reçu")

    results = []

    for file in files:
        filename  = file.filename.lower()
        mime_type = file.content_type or ""
        is_pdf    = filename.endswith('.pdf')
        is_image  = filename.endswith(('.jpg', '.jpeg', '.png', '.webp')) or mime_type.startswith('image/')

        if not is_pdf and not is_image:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Format non supporté (PDF, JPG, PNG uniquement)"
            })
            continue

        try:
            file_bytes = await file.read()
            print(f"📁 Batch: {file.filename} ({len(file_bytes)} bytes)")

            if is_pdf:
                text = extract_text_from_pdf(file_bytes)
                file_type = "pdf"
                if not text:
                    results.append({"filename": file.filename, "success": False, "error": "Impossible d'extraire le texte du PDF"})
                    continue
            else:
                if filename.endswith('.png'):
                    detected_mime = "image/png"
                elif filename.endswith('.webp'):
                    detected_mime = "image/webp"
                else:
                    detected_mime = "image/jpeg"
                text = extract_text_from_image(file_bytes, detected_mime, file.filename)
                file_type = "image"

            print(f"📝 {file.filename} → {len(text)} chars")

            chunks = split_text_into_chunks(text)
            if not chunks:
                results.append({"filename": file.filename, "success": False, "error": "Document vide ou trop court"})
                continue

            doc_name = file.filename.rsplit('.', 1)[0]
            stored = save_chunks(chunks, doc_name, category, file.filename, file_type)

            print(f"✅ {file.filename} → {stored} chunks")
            results.append({
                "filename": file.filename,
                "success": True,
                "chunks": stored,
                "file_type": file_type
            })

        except Exception as e:
            print(f"❌ {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    total_ok    = sum(1 for r in results if r.get("success"))
    total_err   = sum(1 for r in results if not r.get("success"))
    total_chunks = sum(r.get("chunks", 0) for r in results)

    return {
        "success": True,
        "total_files": len(files),
        "total_ok": total_ok,
        "total_errors": total_err,
        "total_chunks": total_chunks,
        "results": results
    }
