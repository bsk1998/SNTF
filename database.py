import asyncpg
import os

# ═══════════════════════════════════════════════════════════
# POOL ASYNCPG — une seule instance partagée pour toute l'app
# Créé au démarrage via @app.on_event("startup") dans main.py
# Détruit proprement via @app.on_event("shutdown")
#
# POURQUOI asyncpg ?
#   psycopg2 est synchrone : chaque get_db() bloquait l'event
#   loop de FastAPI, empêchant le traitement des autres requêtes
#   pendant l'attente I/O base de données.
#   asyncpg est 100% async : le serveur continue à traiter
#   d'autres requêtes pendant les opérations DB.
#
# POURQUOI un pool ?
#   Sans pool : 1 connexion ouverte + fermée par requête HTTP
#   → Supabase refuse les connexions au-delà de ~20 simultanées
#   Avec pool (min=2, max=10) : les connexions sont réutilisées
# ═══════════════════════════════════════════════════════════

_pool: asyncpg.Pool = None


async def create_pool() -> asyncpg.Pool:
    """Initialise le pool au démarrage de l'application."""
    global _pool
    _pool = await asyncpg.create_pool(
        host=os.environ.get("DB_HOST", "aws-0-eu-central-1.pooler.supabase.com"),
        database=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=int(os.environ.get("DB_PORT", "5432")),
        ssl="require",
        min_size=2,      # connexions maintenues ouvertes en permanence
        max_size=10,     # maximum de connexions simultanées
        command_timeout=30,
        statement_cache_size=0,  # requis pour Supabase PgBouncer
    )
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Retourne le pool existant (ou le crée si absent)."""
    global _pool
    if _pool is None:
        await create_pool()
    return _pool


async def close_pool():
    """Ferme proprement toutes les connexions à l'arrêt."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
