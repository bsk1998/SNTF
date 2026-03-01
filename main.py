from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, chat, documents

app = FastAPI(title="SNTF Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])

@app.get("/")
def root():
    return {"status": "✅ SNTF Assistant API v2.0 is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}
