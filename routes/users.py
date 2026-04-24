from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, hashlib

router = APIRouter()

# ═══════════════════════════════════════════════════════════
# VÉRIFICATION ACCÈS ADMIN
# ═══════════════════════════════════════════════════════════
def verify_admin_access(admin_key: str) -> bool:
    if not admin_key:
        raise HTTPException(status_code=403, detail="Clé admin incorrecte")

    ADMIN_KEY = os.environ.get("ADMIN_KEY", "sntf_admin_2024")

    if admin_key == ADMIN_KEY:
        return True

    if admin_key.startswith("eyJ"):
        try:
            from jose import JWTError, jwt
            secret = os.environ.get("JWT_SECRET", "")
            if not secret:
                secret = hashlib.sha256((ADMIN_KEY + "jwt_salt").encode()).hexdigest()
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
# HELPERS DB — async avec asyncpg
#
# CORRECTION : ensure_tables() retiré des routes.
# Les tables sont créées au startup dans main.py.
# ═══════════════════════════════════════════════════════════
async def get_pool():
    from database import get_pool as _get_pool
    return await _get_pool()

async def get_config(key: str, default: str = "") -> str:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT value FROM sntf_config WHERE key = $1", key)
        return val if val is not None else default
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════
# REGISTER / LOGIN — async avec asyncpg
# ═══════════════════════════════════════════════════════════
@router.post("/register")
async def register_user(data: UserInfo):
    try:
        auto_approve = await get_config("auto_approve", "true") == "true"
        pool = await get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT status FROM sntf_users WHERE email = $1", data.email
            )
            if existing:
                status = existing["status"]
                await conn.execute(
                    "UPDATE sntf_users SET last_login = NOW(), login_count = login_count + 1, display_name = $1 WHERE email = $2",
                    data.display_name, data.email
                )
            else:
                status = "approved" if auto_approve else "pending"
                await conn.execute(
                    "INSERT INTO sntf_users (email, display_name, provider, status, last_login, login_count) VALUES ($1, $2, $3, $4, NOW(), 1)",
                    data.email, data.display_name, data.provider, status
                )
        return {"success": True, "status": status, "auto_approve": auto_approve}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
async def check_user(data: UserInfo):
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT status FROM sntf_users WHERE email = $1", data.email)
        if not row:
            return await register_user(data)
        return {"success": True, "status": row["status"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ADMIN — LISTE UTILISATEURS
# ═══════════════════════════════════════════════════════════
@router.post("/list")
async def list_users(data: dict):
    verify_admin_access(data.get("admin_key", ""))
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT email, display_name, provider, status,
                       created_at, last_login, login_count
                FROM sntf_users
                ORDER BY
                    CASE status WHEN 'pending' THEN 0 WHEN 'approved' THEN 1 ELSE 2 END,
                    created_at DESC
            """)

        users = [
            {
                "email":        r["email"],
                "display_name": r["display_name"] or "",
                "provider":     r["provider"] or "google",
                "status":       r["status"],
                "created_at":   r["created_at"].isoformat() if r["created_at"] else "",
                "last_login":   r["last_login"].isoformat() if r["last_login"] else "",
                "login_count":  r["login_count"] or 0
            }
            for r in rows
        ]

        stats = {
            "total":    len(users),
            "pending":  sum(1 for u in users if u["status"] == "pending"),
            "approved": sum(1 for u in users if u["status"] == "approved"),
            "blocked":  sum(1 for u in users if u["status"] == "blocked"),
        }

        return {"success": True, "users": users, "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ADMIN — ACTIONS SUR UTILISATEURS
# ═══════════════════════════════════════════════════════════
@router.post("/approve")
async def approve_user(data: AdminAction):
    verify_admin_access(data.admin_key)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE sntf_users SET status = 'approved' WHERE email = $1", data.email)
        return {"success": True, "message": f"{data.email} approuvé"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/block")
async def block_user(data: AdminAction):
    verify_admin_access(data.admin_key)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE sntf_users SET status = 'blocked' WHERE email = $1", data.email)
        return {"success": True, "message": f"{data.email} bloqué"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
async def delete_user(data: AdminAction):
    verify_admin_access(data.admin_key)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sntf_users WHERE email = $1", data.email)
        return {"success": True, "message": f"{data.email} supprimé"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# ADMIN — CONFIGURATION
# ═══════════════════════════════════════════════════════════
@router.post("/config")
async def update_config(data: ConfigUpdate):
    verify_admin_access(data.admin_key)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sntf_config (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
            """, data.key, data.value)
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/get")
async def get_config_route(data: dict):
    verify_admin_access(data.get("admin_key", ""))
    try:
        val = await get_config(data.get("key", "auto_approve"), "true")
        return {"success": True, "value": val}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
