from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.documents import router as documents_router
from routes.users import router as users_router

app = FastAPI(title="SNTF Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(users_router, prefix="/api/users", tags=["Users"])

# Servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/upload.html")
def upload_page():
    return FileResponse("static/upload.html")

@app.get("/health")
def health():
    return {"status": "healthy"}
