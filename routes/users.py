from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, datetime

router = APIRouter()

# ═══════════════════════════════════════════════════════════
# HELPER CENTRAL — vérifie l'accès admin
# Accepte SOIT la vraie clé ADMIN_KEY (ancien système)
# SOIT un JWT valide signé par /api/admin-auth/login (nouveau)
# ═══════════════════════════════════════════════════════════
def verify_admin_access(admin_key: str) -> bool:
    """
    Retourne True si admin_key est valide.
    Lève HTTPException 403 sinon.
    """
    if not admin_key:
        raise HTTPException(status_code=403, detail="Clé admin incorrecte")

    ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")

    # 1. Vérification directe de la clé (ancien système / upload.html)
    if admin_key == ADMIN_KEY:
        return True

    # 2. Vérification JWT (nouveau système sécurisé)
    if admin_key.startswith("eyJ"):
        try:
            from jose import JWTError, jwt
            import hashlib

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


# ═══════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════
class UserInfo(BaseModel):
    email: str
    display_name: str = ""
    provider: str = "google"

class AdminAction(BaseModel):
    admin_key: str
    email: str

class ConfigUpdate(BaseModel):
    admin_key: str
    key: str
    value: str


# ═══════════════════════════════════════════════════════════
# HELPERS DB
# ═══════════════════════════════════════════════════════════
def get_conn():
    from database import get_db
    return get_db()

def ensure_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sntf_users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            display_name TEXT DEFAULT '',
            provider TEXT DEFAULT 'google',
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW(),
            last_login TIMESTAMP,
            login_count INTEGER DEFAULT 0
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sntf_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        INSERT INTO sntf_config (key, value)
        VALUES ('auto_approve', 'true')
        ON CONFLICT (key) DO NOTHING;
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_config(key: str, default: str = "") -> str:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT value FROM sntf_config WHERE key = %s", (key,))
        row = cur.fetchone()
        cur.close(); conn.close()
        return row[0] if row else default
    except:
        return default


# ═══════════════════════════════════════════════════════════
# ROUTE : Enregistrer / mettre à jour un user à la connexion
# ═══════════════════════════════════════════════════════════
@router.post("/register")
async def register_user(data: UserInfo):
    try:
        ensure_tables()
        auto_approve = get_config("auto_approve", "true") == "true"

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT status FROM sntf_users WHERE email = %s", (data.email,))
        existing = cur.fetchone()

        if existing:
            status = existing[0]
            cur.execute("""
                UPDATE sntf_users
                SET last_login = NOW(), login_count = login_count + 1,
                    display_name = %s
                WHERE email = %s
            """, (data.display_name, data.email))
            conn.commit()
        else:
            status = "approved" if auto_approve else "pending"
            cur.execute("""
                INSERT INTO sntf_users (email, display_name, provider, status, last_login, login_count)
                VALUES (%s, %s, %s, %s, NOW(), 1)
            """, (data.email, data.display_name, data.provider, status))
            conn.commit()

        cur.close(); conn.close()
        return {"success": True, "status": status, "auto_approve": auto_approve}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE : Vérifier le statut d'un user
# ═══════════════════════════════════════════════════════════
@router.post("/check")
async def check_user(data: UserInfo):
    try:
        ensure_tables()
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT status FROM sntf_users WHERE email = %s", (data.email,))
        row = cur.fetchone()
        cur.close(); conn.close()

        if not row:
            return await register_user(data)

        return {"success": True, "status": row[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE ADMIN : Liste tous les users
# ═══════════════════════════════════════════════════════════
@router.post("/list")
async def list_users(data: dict):
    verify_admin_access(data.get("admin_key", ""))
    try:
        ensure_tables()
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT email, display_name, provider, status,
                   created_at, last_login, login_count
            FROM sntf_users
            ORDER BY
                CASE status WHEN 'pending' THEN 0 WHEN 'approved' THEN 1 ELSE 2 END,
                created_at DESC
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()

        users = []
        for r in rows:
            users.append({
                "email": r[0],
                "display_name": r[1] or "",
                "provider": r[2] or "google",
                "status": r[3],
                "created_at": r[4].isoformat() if r[4] else "",
                "last_login": r[5].isoformat() if r[5] else "",
                "login_count": r[6] or 0
            })

        stats = {
            "total": len(users),
            "pending": sum(1 for u in users if u["status"] == "pending"),
            "approved": sum(1 for u in users if u["status"] == "approved"),
            "blocked": sum(1 for u in users if u["status"] == "blocked"),
        }

        return {"success": True, "users": users, "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE ADMIN : Approuver un user
# ═══════════════════════════════════════════════════════════
@router.post("/approve")
async def approve_user(data: AdminAction):
    verify_admin_access(data.admin_key)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE sntf_users SET status = 'approved' WHERE email = %s",
            (data.email,)
        )
        conn.commit()
        cur.close(); conn.close()
        return {"success": True, "message": f"{data.email} approuvé"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE ADMIN : Bloquer un user
# ═══════════════════════════════════════════════════════════
@router.post("/block")
async def block_user(data: AdminAction):
    verify_admin_access(data.admin_key)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE sntf_users SET status = 'blocked' WHERE email = %s",
            (data.email,)
        )
        conn.commit()
        cur.close(); conn.close()
        return {"success": True, "message": f"{data.email} bloqué"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE ADMIN : Supprimer un user
# ═══════════════════════════════════════════════════════════
@router.post("/delete")
async def delete_user(data: AdminAction):
    verify_admin_access(data.admin_key)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM sntf_users WHERE email = %s", (data.email,))
        conn.commit()
        cur.close(); conn.close()
        return {"success": True, "message": f"{data.email} supprimé"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE ADMIN : Changer auto_approve
# ═══════════════════════════════════════════════════════════
@router.post("/config")
async def update_config(data: ConfigUpdate):
    verify_admin_access(data.admin_key)
    try:
        ensure_tables()
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sntf_config (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
        """, (data.key, data.value, data.value))
        conn.commit()
        cur.close(); conn.close()
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ROUTE ADMIN : Lire auto_approve
# ═══════════════════════════════════════════════════════════
@router.post("/config/get")
async def get_config_route(data: dict):
    verify_admin_access(data.get("admin_key", ""))
    try:
        ensure_tables()
        val = get_config(data.get("key", "auto_approve"), "true")
        return {"success": True, "value": val}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
