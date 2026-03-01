import psycopg2
import os

def get_db():
    """
    Connexion à Supabase - même base que n8n
    Host: aws-1-eu-central-1.pooler.supabase.com (Session Pooler IPv4)
    """
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "aws-1-eu-central-1.pooler.supabase.com"),
        database=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres.zlnkagxpbqjcaumdttnt"),
        password=os.environ.get("DB_PASSWORD"),
        port=int(os.environ.get("DB_PORT", "5432")),
        sslmode="require"
    )
    return conn
