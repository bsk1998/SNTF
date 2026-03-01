from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_db
import hashlib
import secrets
import base64
import json
import time

router = APIRouter()

class RegisterRequest(BaseModel):
    email: str
    password: str
    bot_id: int = 1

class LoginRequest(BaseModel):
    email: str
    password: str

def hash_password(password: str, salt: str) -> str:
    """Même algorithme que le nœud 'Chiffrer Mot de Passe' dans n8n"""
    return hashlib.pbkdf2_hmac(
        'sha512',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        10000
    ).hex()

def generate_token(user_id: int, email: str, role: str, bot_id: int) -> str:
    """Même format que le nœud 'Générer Token' dans n8n"""
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "bot_id": bot_id,
        "exp": int(time.time()) + (60 * 60 * 24 * 7)  # 7 jours
    }
    return "sntf_" + base64.b64encode(json.dumps(payload).encode()).decode()

@router.post("/register")
def register(request: RegisterRequest):
    """
    Equivalent du workflow SNTF_AUTH - Flux Inscription :
    Webhook → Valider → Chiffrer MDP → Vérifier email → Créer compte
    """
    email = request.email.lower().strip()
    password = request.password

    if not email or not password:
        raise HTTPException(400, "Email et mot de passe requis")
    if len(password) < 6:
        raise HTTPException(400, "Le mot de passe doit avoir au moins 6 caractères")
    if "@" not in email:
        raise HTTPException(400, "Email invalide")

    conn = get_db()
    cur = conn.cursor()
    try:
        # Vérifier si email existe déjà
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            raise HTTPException(409, "Cet email est déjà utilisé")

        # Chiffrer le mot de passe
        salt = secrets.token_hex(16)
        password_hash = hash_password(password, salt)

        # Créer le compte
        cur.execute(
            """INSERT INTO users (email, password_hash, salt, bot_id, role, created_at)
               VALUES (%s, %s, %s, %s, 'user', NOW()) RETURNING id""",
            (email, password_hash, salt, request.bot_id)
        )
        user_id = cur.fetchone()[0]
        conn.commit()

        return {
            "success": True,
            "message": "Compte créé avec succès ! Bienvenue dans l'Assistant SNTF.",
            "user_id": user_id
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, f"Erreur serveur: {str(e)}")
    finally:
        cur.close()
        conn.close()

@router.post("/login")
def login(request: LoginRequest):
    """
    Equivalent du workflow SNTF_AUTH - Flux Connexion :
    Webhook → Trouver utilisateur → Vérifier MDP → Générer Token
    """
    email = request.email.lower().strip()

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, email, password_hash, salt, role, bot_id FROM users WHERE email = %s",
            (email,)
        )
        user = cur.fetchone()

        if not user:
            raise HTTPException(401, "Email ou mot de passe incorrect")

        user_id, user_email, stored_hash, salt, role, bot_id = user

        # Vérifier le mot de passe
        if hash_password(request.password, salt) != stored_hash:
            raise HTTPException(401, "Email ou mot de passe incorrect")

        token = generate_token(user_id, user_email, role, bot_id)

        return {
            "success": True,
            "message": "Connexion réussie ! Bienvenue.",
            "token": token,
            "user": {
                "id": user_id,
                "email": user_email,
                "role": role,
                "bot_id": bot_id
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur serveur: {str(e)}")
    finally:
        cur.close()
        conn.close()
