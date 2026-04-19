from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.documents import router as documents_router
from routes.users import router as users_router
from routes.admin_auth import router as admin_auth_router
import os
import time

app = FastAPI(title="SNTF Assistant API", version="2.0.0")

# ═══════════════════════════════════════════════════════════
# CORS
# ═══════════════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# MIDDLEWARE SÉCURITÉ — headers HTTP sécurisés sur toutes
# les réponses (protection XSS, clickjacking, sniffing)
# ═══════════════════════════════════════════════════════════
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    process_time = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time"] = f"{process_time}ms"
    return response

# ═══════════════════════════════════════════════════════════
# GESTIONNAIRE D'ERREURS GLOBAL
# ═══════════════════════════════════════════════════════════
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    return JSONResponse(
        status_code=429,
        content={"success": False, "detail": str(exc.detail)},
        headers={"Retry-After": "60"}
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Erreur serveur interne. Réessayez."}
    )

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════
app.include_router(auth_router,       prefix="/api/auth",       tags=["Auth"])
app.include_router(chat_router,       prefix="/api/chat",       tags=["Chat"])
app.include_router(documents_router,  prefix="/api/documents",  tags=["Documents"])
app.include_router(users_router,      prefix="/api/users",      tags=["Users"])
app.include_router(admin_auth_router, prefix="/api/admin-auth", tags=["AdminAuth"])

# ═══════════════════════════════════════════════════════════
# FICHIERS STATIQUES
# ═══════════════════════════════════════════════════════════
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")

@app.get("/upload.html")
def upload_page():
    return FileResponse("static/upload.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
      <rect width="32" height="32" rx="6" fill="#1565C0"/>
      <text x="16" y="22" font-family="Arial" font-size="16" font-weight="bold"
            text-anchor="middle" fill="white">S</text>
    </svg>"""
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
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
            "groq_key": bool(os.environ.get("GROQ_API_KEY")),
            "hf_key": bool(os.environ.get("HF_API_KEY")),
            "admin_key_set": bool(os.environ.get("ADMIN_KEY")),
            "jwt_secret_set": bool(os.environ.get("JWT_SECRET"))
        }
    except Exception as e:
        return {"status": "healthy", "db_error": str(e)[:100]}
