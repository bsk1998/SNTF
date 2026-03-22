from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.documents import router as documents_router
from routes.users import router as users_router
import os

app = FastAPI(title="SNTF Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,      prefix="/api/auth",      tags=["Auth"])
app.include_router(chat_router,      prefix="/api/chat",      tags=["Chat"])
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(users_router,     prefix="/api/users",     tags=["Users"])

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/upload.html")
def upload_page():
    return FileResponse("static/upload.html")

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Endpoint de santé — utilisé par UptimeRobot pour garder le service éveillé"""
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        doc_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {
            "status": "healthy",
            "documents": doc_count,
            "groq_key": bool(os.environ.get("GROQ_API_KEY"))
        }
    except Exception as e:
        return {"status": "healthy", "db_error": str(e)[:100]}
