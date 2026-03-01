# 🚂 SNTF Assistant API v2.0
## Serveur Python — Remplacement de n8n

Ce serveur reproduit EXACTEMENT le workflow n8n CHATBOOT_SNTF 2.0.

---

## Correspondance n8n → Python

| Nœud n8n | Endpoint Python |
|----------|----------------|
| 💬 Interface Chat | POST `/api/chat/message` |
| 🤖 Assistant SNTF (Groq llama-3.3-70b) | Intégré dans chat.py |
| 📚 Recherche Documents SNTF | Intégré dans chat.py |
| 📋 Formulaire Upload PDF | POST `/api/documents/upload` |
| 📄 Extraction Texte PDF | Intégré dans documents.py |
| ✂️ Découpage en Chunks (512/50) | Intégré dans documents.py |
| 💾 Stocker dans Supabase (n8n_vectors) | Intégré dans documents.py |
| 📝 Webhook Inscription | POST `/api/auth/register` |
| 🔐 Webhook Connexion | POST `/api/auth/login` |

---

## Variables d'environnement (Render.com)

| Variable | Description |
|----------|-------------|
| DB_HOST | aws-1-eu-central-1.pooler.supabase.com |
| DB_NAME | postgres |
| DB_USER | postgres.zlnkagxpbqjcaumdttnt |
| DB_PORT | 5432 |
| DB_PASSWORD | Ton mot de passe Supabase |
| GROQ_API_KEY | Ta clé API Groq (gsk_...) |
| HF_API_KEY | Ta clé API Hugging Face (hf_...) |
| ADMIN_KEY | Mot de passe secret pour uploader les PDF |

---

## Déploiement

1. Upload ce dossier sur GitHub
2. Connecte GitHub à Render.com
3. New Web Service → sélectionne le repo
4. Configure les variables d'environnement
5. Deploy !

---

## Test local

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# API disponible sur http://localhost:8000
# Documentation sur http://localhost:8000/docs
```
