import os
import sys
import requests
import numpy as np
from pypdf import PdfReader
import psycopg2

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Supabase/Postgres config
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432")

VECTOR_TABLE = "n8n_vectors"

def get_embedding(text: str) -> list:
    """Génère l'embedding avec HuggingFace"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        if response.status_code == 200:
            embedding = response.json()
            if isinstance(embedding[0], list):
                embedding = np.mean(embedding, axis=0).tolist()
            return embedding
        else:
            print(f"Erreur HuggingFace: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Erreur embedding: {e}")
    return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Erreur lecture PDF: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    """Découpe le texte en chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def upload_to_supabase(chunks: list, pdf_name: str):
    """Upload les chunks avec embeddings dans Supabase"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            embedding = get_embedding(chunk)
            if not embedding:
                print(f"Échec embedding pour chunk {i+1}")
                continue
            
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            metadata = {"source": pdf_name, "chunk": i+1}
            
            cur.execute(
                f"""INSERT INTO {VECTOR_TABLE} (content, metadata, embedding)
                   VALUES (%s, %s, %s::vector)""",
                (chunk, str(metadata), embedding_str)
            )
        
        conn.commit()
        print(f"✅ {len(chunks)} chunks uploadés avec succès!")
        
    except Exception as e:
        print(f"Erreur Supabase: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_pdf.py <chemin_vers_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Fichier introuvable: {pdf_path}")
        sys.exit(1)
    
    print(f"📄 Extraction du texte de {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("Aucun texte extrait du PDF")
        sys.exit(1)
    
    print(f"✂️ Découpage en chunks...")
    chunks = split_text_into_chunks(text)
    print(f"   {len(chunks)} chunks créés")
    
    print(f"⬆️ Upload vers Supabase...")
    upload_to_supabase(chunks, os.path.basename(pdf_path))
    
    print("🎉 Terminé!")

if __name__ == "__main__":
    main()
