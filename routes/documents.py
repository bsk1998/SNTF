from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
import io, os, json, hashlib, math, requests, base64, threading

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VECTOR_TABLE = "document_chunks"

# ═══════════════════════════════════════════════════════════
# VÉRIFICATION ACCÈS ADMIN
# ═══════════════════════════════════════════════════════════
def verify_admin_access(admin_key: str) -> bool:
    if not admin_key:
        raise HTTPException(status_code=403, detail="Clé admin incorrecte")

    ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")

    if admin_key == ADMIN_KEY:
        return True

    if admin_key.startswith("eyJ"):
        try:
            from jose import JWTError, jwt
            secret = os.environ.get("JWT_SECRET", "")
            if not secret:
                secret = hashlib.sha256((ADMIN_KEY + "jwt_salt").encode()).hexdigest()
            payload = jwt.decode(admin_key, secret, algorithms=["HS256"])
            if payload.get("role") == "admin":
                return True
        except Exception:
            raise HTTPException(status_code=403, detail="Session expirée. Reconnectez-vous.")

    raise HTTPException(status_code=403, detail="Clé admin incorrecte")


# ═══════════════════════════════════════════════════════════
# EMBEDDING — _model protégé par un Lock threading
# (même correction que routes/chat.py)
# ═══════════════════════════════════════════════════════════
_model = None
_model_lock = threading.Lock()  # ← CORRECTION : lock ajouté


def get_embedding(text: str) -> list:
    global _model

    # Méthode 1 : modèle local — chargement protégé par Lock
    with _model_lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                _model = False

    if _model:
        try:
            return _model.encode(text[:500]).tolist()
        except Exception:
            pass

    # Méthode 2 : HuggingFace API
    HF_KEY = os.environ.get("HF_API_KEY")
    if HF_KEY:
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_KEY}"},
                json={"inputs": text[:500], "options": {"wait_for_model": True}},
                timeout=60
            )
            if r.status_code == 200:
                emb = r.json()
                if isinstance(emb[0], list):
                    import numpy as np
                    emb = np.mean(emb, axis=0).tolist()
                return emb
        except Exception:
            pass

    # Méthode 3 : fallback hash MD5
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:384]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x / norm for x in vector]


# ═══════════════════════════════════════════════════════════
# EXTRACTION TEXTE PDF
# ═══════════════════════════════════════════════════════════
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    import re

    def fix_spacing(text):
        if not text:
            return ""
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"([a-zàâéèêëîïôùûü])([A-ZÀÂÉÈÊËÎÏÔÙÛÜ])", r"\1 \2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = text.split("\n")
        clean = []
        for line in lines:
            s = line.strip()
            if not s:
                clean.append("")
                continue
            words = s.split()
            if len(words) >= 3 or (len(words) >= 1 and s[0].isupper() and len(s) >= 5):
                clean.append(s)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(clean)).strip()

    def quality(text):
        if not text or len(text) < 50:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        correct = [w for w in words if len(w) >= 3 and sum(c.isalpha() for c in w) >= len(w) * 0.6]
        ratio = len(correct) / len(words)
        french = ["les", "des", "est", "que", "pour", "une", "dans", "avec", "sur", "par"]
        bonus = sum(1 for w in french if w in text.lower()) * 0.015
        return min(ratio + bonus, 1.0)

    best_text = ""
    best_score = 0.0

    try:
        import pdfplumber
        for tol in [1, 2, 3, 5]:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    pages = []
                    for page in pdf.pages:
                        t = page.extract_text(x_tolerance=tol, y_tolerance=tol)
                        if t:
                            pages.append(t)
                    text = fix_spacing("\n\n".join(pages))
                    q = quality(text)
                    if q > best_score and len(text) >= 100:
                        best_score = q
                        best_text = text
                    if q >= 0.75:
                        break
            except Exception:
                pass
    except ImportError:
        pass

    if best_score >= 0.65 and len(best_text) >= 100:
        return best_text

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                pages.append(fix_spacing(t))
        text = "\n\n".join(p for p in pages if p)
        q = quality(text)
        if q > best_score and len(text) >= 100:
            best_score = q
            best_text = text
    except Exception:
        pass

    if best_score >= 0.60 and len(best_text) >= 100:
        return best_text

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            import fitz
            doc_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for i in range(min(len(doc_pdf), 20)):
                page = doc_pdf[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB)
                img_b64 = base64.b64encode(pix.tobytes("jpeg")).decode()
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": "Transcris INTEGRALEMENT tout le texte de cette page. Respecte les titres et paragraphes. Transcris mot pour mot sans reformuler."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]}],
                        "max_tokens": 2048
                    },
                    timeout=60
                )
                if r.status_code == 200:
                    pages_text.append(r.json()["choices"][0]["message"]["content"])
            if pages_text:
                return "\n\n".join(pages_text)
        except Exception:
            pass

    return best_text


# ═══════════════════════════════════════════════════════════
# EXTRACTION TEXTE IMAGE
# ═══════════════════════════════════════════════════════════
def extract_text_from_image(image_bytes: bytes, mime_type: str, filename: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY manquante pour analyser les images")
    try:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Analyse cette image en détail. Extrais et décris : 1. Tout le texte visible 2. Le contenu principal 3. Les informations techniques. Réponds en français, commence par : 'Cette image montre...'"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                ]}],
                "max_tokens": 2048
            },
            timeout=60
        )
        if r.status_code == 200:
            return f"[Image: {filename}]\n\n{r.json()['choices'][0]['message']['content']}"
        else:
            raise HTTPException(500, f"Erreur analyse image: {r.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse image: {str(e)}")


# ═══════════════════════════════════════════════════════════
# DÉCOUPAGE EN CHUNKS
# ═══════════════════════════════════════════════════════════
def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current and len(current.strip()) >= 50:
                        chunks.append(current.strip())
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


# ═══════════════════════════════════════════════════════════
# SAUVEGARDER CHUNKS — async avec asyncpg
# ═══════════════════════════════════════════════════════════
async def save_chunks(chunks: list, doc_name: str, category: str, source_filename: str, file_type: str) -> int:
    from database import get_pool
    pool = await get_pool()
    stored = 0
    async with pool.acquire() as conn:
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
            await conn.execute(
                f"INSERT INTO {VECTOR_TABLE} (content, metadata, embedding) VALUES ($1, $2::jsonb, $3::vector)",
                chunk, metadata, emb_str
            )
            stored += 1
    return stored


# ═══════════════════════════════════════════════════════════
# UPLOAD — PDF + IMAGES
# ═══════════════════════════════════════════════════════════
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    document_name: str = Form(default=""),
    category: str = Form(default="General"),
    admin_key: str = Form(default="")
):
    verify_admin_access(admin_key)

    filename  = file.filename.lower()
    mime_type = file.content_type or ""
    is_pdf    = filename.endswith('.pdf')
    is_image  = filename.endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')) or mime_type.startswith('image/')

    if not is_pdf and not is_image:
        raise HTTPException(400, "Types acceptés : PDF, JPG, PNG, WEBP")

    try:
        file_bytes = await file.read()

        if is_pdf:
            text = extract_text_from_pdf(file_bytes)
            file_type = "pdf"
            if not text:
                raise HTTPException(400, "Impossible d'extraire le texte du PDF")
        else:
            detected_mime = "image/png" if filename.endswith('.png') else ("image/webp" if filename.endswith('.webp') else "image/jpeg")
            text = extract_text_from_image(file_bytes, detected_mime, file.filename)
            file_type = "image"

        chunks = split_text_into_chunks(text)
        if not chunks:
            raise HTTPException(400, "Document trop court ou vide")

        doc_name = document_name if document_name else file.filename.rsplit('.', 1)[0]
        stored = await save_chunks(chunks, doc_name, category, file.filename, file_type)

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
        raise HTTPException(500, f"Erreur: {str(e)}")


# ═══════════════════════════════════════════════════════════
# LIST — async avec asyncpg
# ═══════════════════════════════════════════════════════════
@router.get("/list")
async def list_documents(admin_key: str = ""):
    verify_admin_access(admin_key)
    try:
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT metadata->>'filename'  AS filename,
                       metadata->>'category'  AS category,
                       metadata->>'file_type' AS file_type,
                       COUNT(*) AS chunks
                FROM {VECTOR_TABLE}
                WHERE metadata->>'filename' IS NOT NULL
                GROUP BY metadata->>'filename', metadata->>'category', metadata->>'file_type'
                ORDER BY metadata->>'filename'
            """)
        return {
            "success":         True,
            "total_documents": len(rows),
            "documents": [
                {"filename": r["filename"], "category": r["category"],
                 "file_type": r["file_type"] or "pdf", "chunks": int(r["chunks"])}
                for r in rows
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# DELETE — async avec asyncpg
# ═══════════════════════════════════════════════════════════
class DeleteRequest(BaseModel):
    filename: str
    admin_key: str = ""

@router.post("/delete")
async def delete_document(req: DeleteRequest):
    verify_admin_access(req.admin_key)
    try:
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {VECTOR_TABLE} WHERE metadata->>'filename' = $1",
                req.filename
            )
        # asyncpg retourne "DELETE N" comme string
        deleted = int(result.split(" ")[-1]) if result else 0
        return {"success": True, "chunks_deleted": deleted, "filename": req.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════
# UPLOAD BATCH — async avec asyncpg
# ═══════════════════════════════════════════════════════════
@router.post("/upload-batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    category: str = Form(default="General"),
    admin_key: str = Form(default="")
):
    verify_admin_access(admin_key)
    if not files:
        raise HTTPException(400, "Aucun fichier reçu")

    results = []
    for file in files:
        filename  = file.filename.lower()
        mime_type = file.content_type or ""
        is_pdf    = filename.endswith('.pdf')
        is_image  = filename.endswith(('.jpg', '.jpeg', '.png', '.webp')) or mime_type.startswith('image/')

        if not is_pdf and not is_image:
            results.append({"filename": file.filename, "success": False, "error": "Format non supporté"})
            continue

        try:
            file_bytes = await file.read()
            if is_pdf:
                text = extract_text_from_pdf(file_bytes)
                file_type = "pdf"
                if not text:
                    results.append({"filename": file.filename, "success": False, "error": "Impossible d'extraire le texte"})
                    continue
            else:
                detected_mime = "image/png" if filename.endswith('.png') else ("image/webp" if filename.endswith('.webp') else "image/jpeg")
                text = extract_text_from_image(file_bytes, detected_mime, file.filename)
                file_type = "image"

            chunks = split_text_into_chunks(text)
            if not chunks:
                results.append({"filename": file.filename, "success": False, "error": "Document vide ou trop court"})
                continue

            doc_name = file.filename.rsplit('.', 1)[0]
            stored = await save_chunks(chunks, doc_name, category, file.filename, file_type)
            results.append({"filename": file.filename, "success": True, "chunks": stored, "file_type": file_type})

        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})

    total_ok     = sum(1 for r in results if r.get("success"))
    total_err    = sum(1 for r in results if not r.get("success"))
    total_chunks = sum(r.get("chunks", 0) for r in results)

    return {
        "success":      True,
        "total_files":  len(files),
        "total_ok":     total_ok,
        "total_errors": total_err,
        "total_chunks": total_chunks,
        "results":      results
    }
