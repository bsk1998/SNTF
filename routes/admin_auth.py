from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
import hashlib
import hmac

router = APIRouter()
security = HTTPBearer()

# ═══════════════════════════════════════════════════════════
# CONFIG — toutes les valeurs viennent des variables Render
# La clé admin n'est JAMAIS dans le code HTML
# ═══════════════════════════════════════════════════════════
def get_admin_key() -> str:
    key = os.environ.get("ADMIN_KEY", "")
    if not key:
        raise HTTPException(500, "ADMIN_KEY non configurée sur le serveur")
    return key

def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        # Fallback : dériver depuis ADMIN_KEY si JWT_SECRET pas défini
        secret = hashlib.sha256(
            (os.environ.get("ADMIN_KEY", "fallback") + "jwt_salt").encode()
        ).hexdigest()
    return secret

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8  # Token valide 8 heures

# ═══════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════
class AdminLoginRequest(BaseModel):
    key: str

class TokenResponse(BaseModel):
    success: bool
    token: str
    expires_in: int  # secondes

# ═══════════════════════════════════════════════════════════
# GÉNÉRATION ET VÉRIFICATION DU TOKEN JWT
# ═══════════════════════════════════════════════════════════
def create_admin_token() -> str:
    """Génère un JWT signé côté serveur avec expiration."""
    secret = get_jwt_secret()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": "admin",
        "role": "admin",
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)

def verify_admin_token(token: str) -> bool:
    """Vérifie la signature et l'expiration du JWT."""
    try:
        secret = get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload.get("role") == "admin"
    except JWTError:
        return False

def require_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dépendance FastAPI — à utiliser sur toutes les routes admin sensibles.
    Usage : @router.get("/route") async def route(admin = Depends(require_admin))
    """
    token = credentials.credentials
    if not verify_admin_token(token):
        raise HTTPException(
            status_code=401,
            detail="Token invalide ou expiré. Reconnectez-vous.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True

# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════
@router.post("/login")
def admin_login(request: AdminLoginRequest):
    """
    Vérifie la clé admin côté serveur (jamais côté client)
    et retourne un JWT signé si correct.
    La comparaison utilise hmac.compare_digest pour éviter
    les attaques timing.
    """
    admin_key = get_admin_key()

    # Comparaison sécurisée contre les attaques timing
    key_valid = hmac.compare_digest(
        request.key.encode("utf-8"),
        admin_key.encode("utf-8")
    )

    if not key_valid:
        # Délai fixe pour ne pas révéler si la clé est proche ou loin
        import time
        time.sleep(0.5)
        raise HTTPException(401, "Clé incorrecte")

    token = create_admin_token()
    return TokenResponse(
        success=True,
        token=token,
        expires_in=JWT_EXPIRE_HOURS * 3600
    )

@router.get("/verify")
def admin_verify(admin=Depends(require_admin)):
    """Vérifie si le token est encore valide. Utilisé par le frontend au chargement."""
    return {"valid": True}
