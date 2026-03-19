import psycopg2
import os

def get_db():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "aws-0-eu-central-1.pooler.supabase.com"),
        database=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=int(os.environ.get("DB_PORT", "5432")),
        sslmode="require",
        connect_timeout=10
    )
    return conn
