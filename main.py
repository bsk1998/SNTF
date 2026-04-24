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
# STARTUP / SHUTDOWN — pool asyncpg + tables DB
# ═══════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    """
    Initialise le pool de connexions une seule fois au démarrage.
    Crée les tables si elles n'existent pas (remplace l'appel
    ensure_conv_table() qui était appelé à chaque requête chat).
    """
    from database import create_pool
    pool = await create_pool()

    async with pool.acquire() as conn:
        # Table conversations
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_email TEXT,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # Table utilisateurs
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sntf_users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                display_name TEXT DEFAULT '',
                provider TEXT DEFAULT 'google',
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP,
                login_count INTEGER DEFAULT 0
            )
        """)
        # Table configuration
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sntf_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            INSERT INTO sntf_config (key, value)
            VALUES ('auto_approve', 'true')
            ON CONFLICT (key) DO NOTHING
        """)

    print("✅ Pool asyncpg initialisé — tables vérifiées")


@app.on_event("shutdown")
async def shutdown():
    """Ferme proprement le pool à l'arrêt du serveur."""
    from database import close_pool
    await close_pool()
    print("🔌 Pool asyncpg fermé")


# ═══════════════════════════════════════════════════════════
# CORS — origines autorisées depuis variable d'environnement
#
# EN PRODUCTION : définir ALLOWED_ORIGINS sur Render.com
#   Ex: ALLOWED_ORIGINS=https://sntf-assistant.onrender.com
#
# EN DÉVELOPPEMENT : laisser vide → accepte tout (localhost)
# ═══════════════════════════════════════════════════════════
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
if _raw_origins.strip():
    # Production : origines explicitement listées
    allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    print(f"🔒 CORS restreint à : {allowed_origins}")
else:
    # Développement ou Render sans variable définie
    allowed_origins = ["*"]
    print("⚠️  CORS ouvert (*) — définir ALLOWED_ORIGINS en production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ═══════════════════════════════════════════════════════════
# MIDDLEWARE SÉCURITÉ — headers HTTP sur toutes les réponses
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
async def health():
    try:
        from database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
        return {
            "status": "healthy",
            "documents": doc_count,
            "groq_key": bool(os.environ.get("GROQ_API_KEY")),
            "hf_key": bool(os.environ.get("HF_API_KEY")),
            "admin_key_set": bool(os.environ.get("ADMIN_KEY")),
            "jwt_secret_set": bool(os.environ.get("JWT_SECRET")),
            "db": "asyncpg pool"
        }
    except Exception as e:
        return {"status": "healthy", "db_error": str(e)[:100]}
