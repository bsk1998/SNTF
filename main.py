from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.documents import router as documents_router

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

# Servir les fichiers statiques (upload.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/upload.html")
def upload_page():
    return FileResponse("static/upload.html")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"status": "SNTF API en ligne", "version": "2.0.0"}
