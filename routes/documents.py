from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import io, os, json, hashlib, math, requests, base64
from database import get_db

router = APIRouter()

# ═══════════════════════════════════════════════════════════
# HELPER CENTRAL — même logique que users.py
# Accepte SOIT la vraie clé ADMIN_KEY SOIT un JWT valide
# ═══════════════════════════════════════════════════════════
def verify_admin_access(admin_key: str) -> bool:
    if not admin_key:
        raise HTTPException(status_code=403, detail="Clé admin incorrecte")

    ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")

    # 1. Clé directe (upload.html / ancien système)
    if admin_key == ADMIN_KEY:
        return True

    # 2. JWT (nouveau panneau admin sécurisé)
    if admin_key.startswith("eyJ"):
        try:
            from jose import JWTError, jwt

            secret = os.environ.get("JWT_SECRET", "")
            if not secret:
                secret = hashlib.sha256(
                    (ADMIN_KEY + "jwt_salt").encode()
                ).hexdigest()

            payload = jwt.decode(admin_key, secret, algorithms=["HS256"])
            if payload.get("role") == "admin":
                return True
        except Exception:
            raise HTTPException(status_code=403, detail="Session expirée. Reconnectez-vous.")

    raise HTTPException(status_code=403, detail="Clé admin incorrecte")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VECTOR_TABLE = "document_chunks"

# ═══════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════
_model = None

def get_embedding(text: str) -> list:
    global _model

    # Méthode 1 : sentence-transformers local
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        return _model.encode(text[:500]).tolist()
    except:
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
        except:
            pass

    # Méthode 3 : Fallback hash
    words = text.lower().split()
    vector = [0.0] * 384
    for i, word in enumerate(words[:384]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % 384] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x**2 for x in vector)) or 1.0
    return [x/norm for x in vector]


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

    # METHODE 1 : pdfplumber
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
    except Exception:
        pass

    if best_score >= 0.65 and len(best_text) >= 100:
        return best_text

    # METHODE 2 : pypdf
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

    # METHODE 3 : Groq Vision
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

    if best_text:
        return best_text

    return ""


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
                    {"type": "text", "text": """Analyse cette image en détail pour créer une base de connaissances.
Extrais et décris :
1. Tout le texte visible (tableaux, titres, données, labels)
2. Le contenu principal de l'image
3. Les informations techniques ou chiffres importants
4. Le contexte général

Réponds en français, de façon complète et structurée. Commence par : 'Cette image montre...'"""},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                ]}],
                "max_tokens": 2048
            },
            timeout=60
        )
        if r.status_code == 200:
            description = r.json()["choices"][0]["message"]["content"]
            return f"[Image: {filename}]\n\n{description}"
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
# SAUVEGARDER CHUNKS EN BASE
# ═══════════════════════════════════════════════════════════
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
    conn.commit()
    cur.close()
    conn.close()
    return stored


# ═══════════════════════════════════════════════════════════
# ENDPOINT UPLOAD — PDF + IMAGES
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
            if filename.endswith('.png'):
                detected_mime = "image/png"
            elif filename.endswith('.webp'):
                detected_mime = "image/webp"
            else:
                detected_mime = "image/jpeg"
            text = extract_text_from_image(file_bytes, detected_mime, file.filename)
            file_type = "image"

        chunks = split_text_into_chunks(text)
        if not chunks:
            raise HTTPException(400, "Document trop court ou vide")

        doc_name = document_name if document_name else file.filename.rsplit('.', 1)[0]
        stored = save_chunks(chunks, doc_name, category, file.filename, file_type)

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
# LIST
# ═══════════════════════════════════════════════════════════
@router.get("/list")
def list_documents(admin_key: str = ""):
    verify_admin_access(admin_key)
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
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════
class DeleteRequest(BaseModel):
    filename: str
    admin_key: str = ""

@router.post("/delete")
def delete_document(req: DeleteRequest):
    verify_admin_access(req.admin_key)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(f"DELETE FROM {VECTOR_TABLE} WHERE metadata->>'filename' = %s", (req.filename,))
        deleted = cur.rowcount
        conn.commit()
        cur.close(); conn.close()
        return {"success": True, "chunks_deleted": deleted, "filename": req.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════
# UPLOAD BATCH
# ═══════════════════════════════════════════════════════════
from typing import List

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
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Format non supporté (PDF, JPG, PNG uniquement)"
            })
            continue

        try:
            file_bytes = await file.read()

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

            chunks = split_text_into_chunks(text)
            if not chunks:
                results.append({"filename": file.filename, "success": False, "error": "Document vide ou trop court"})
                continue

            doc_name = file.filename.rsplit('.', 1)[0]
            stored = save_chunks(chunks, doc_name, category, file.filename, file_type)

            results.append({
                "filename": file.filename,
                "success": True,
                "chunks": stored,
                "file_type": file_type
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    total_ok     = sum(1 for r in results if r.get("success"))
    total_err    = sum(1 for r in results if not r.get("success"))
    total_chunks = sum(r.get("chunks", 0) for r in results)

    return {
        "success": True,
        "total_files": len(files),
        "total_ok": total_ok,
        "total_errors": total_err,
        "total_chunks": total_chunks,
        "results": results
    }
