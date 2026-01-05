# 🚀 Guide de démarrage - Agent IA BTP

## Prérequis

- Python 3.11 ou supérieur
- pip (gestionnaire de paquets Python)
- Variables d'environnement configurées (voir ci-dessous)

## 📋 Étapes de démarrage

### 1. Se placer dans le dossier du projet

```powershell
cd "C:\Users\caoer\Desktop\Projet AgentAI\agent IA"
```

### 2. Créer un environnement virtuel (recommandé)

```powershell
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1
```

**Note:** Si vous avez une erreur d'exécution de scripts, exécutez d'abord :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Installer les dépendances

```powershell
pip install -r requirements.txt
```

**Dépendances optionnelles (si besoin) :**
- Pour OCR (images) : installer Tesseract OCR séparément
- Pour PDF avancé : WeasyPrint (déjà dans requirements.txt)

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du dossier `agent IA` :

```env
# OpenAI (obligatoire)
OPENAI_API_KEY=sk-votre-cle-api-openai

# Supabase (obligatoire pour RAG et stockage)
SUPABASE_URL=https://votre-projet.supabase.co
SUPABASE_SERVICE_ROLE_KEY=votre-cle-service-role
# OU
SUPABASE_ANON_KEY=votre-cle-anon

# Modèles LLM (optionnel, valeurs par défaut)
LLM_MODEL=gpt-4o-mini
LLM_FALLBACK_MODEL=gpt-4o

# CORS (optionnel)
AI_CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173

# LangSmith (optionnel, pour traçage)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=votre-cle-langsmith
```

### 5. Lancer le serveur FastAPI

```powershell
uvicorn agent.api:app --host 0.0.0.0 --port 8000 --reload
```

**Options utiles :**
- `--reload` : Recharge automatiquement lors des modifications (développement)
- `--host 0.0.0.0` : Accessible depuis d'autres machines sur le réseau
- `--port 8000` : Port par défaut (peut être changé)

### 6. Vérifier que le serveur fonctionne

Ouvrir dans le navigateur :
- **API** : http://localhost:8000
- **Documentation interactive** : http://localhost:8000/docs
- **Documentation alternative** : http://localhost:8000/redoc
- **Interface statique** : http://localhost:8000/static (si disponible)

## 🧪 Tester l'API

### Test rapide avec curl (PowerShell)

```powershell
# Test de l'endpoint chat
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -ContentType "application/json" -Body '{"message": "Bonjour, peux-tu m''aider à créer un devis ?"}'

# Test de l'endpoint root
Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET
```

### Test avec le frontend

Si vous avez le frontend React/Next.js dans `nextmind-bid-builder-main`, assurez-vous qu'il pointe vers `http://localhost:8000` pour les appels API.

## 🐳 Alternative : Docker

Si vous préférez utiliser Docker :

```powershell
# Construire l'image
docker build -t agent-ia-btp .

# Lancer le conteneur
docker run -p 8000:8000 --env-file .env agent-ia-btp
```

## ⚠️ Dépannage

### Erreur : Module non trouvé
```powershell
# Réinstaller les dépendances
pip install -r requirements.txt --upgrade
```

### Erreur : Port déjà utilisé
```powershell
# Utiliser un autre port
uvicorn agent.api:app --host 0.0.0.0 --port 8001 --reload
```

### Erreur : Variables d'environnement non trouvées
- Vérifier que le fichier `.env` est bien à la racine du dossier `agent IA`
- Vérifier que `python-dotenv` est installé (normalement inclus dans les dépendances)

### Erreur : Supabase non configuré
- L'API fonctionnera mais le RAG et le stockage Supabase ne seront pas disponibles
- Les endpoints `/chat` et `/analyze` fonctionneront quand même

## 📝 Commandes utiles

```powershell
# Voir les logs en temps réel
uvicorn agent.api:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Lancer sans reload (production)
uvicorn agent.api:app --host 0.0.0.0 --port 8000

# Lancer avec plusieurs workers (production)
uvicorn agent.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔗 Endpoints disponibles

- `GET /` : Page d'accueil ou statut API
- `POST /chat` : Chat conversationnel avec l'agent
- `POST /analyze` : Analyse de fichiers (PDF/DOCX/images)
- `POST /prepare-devis` : Préparation de devis/factures
- `GET /prepare-devis/prefill` : Pré-remplissage depuis Supabase
- `GET /docs` : Documentation Swagger interactive
- `GET /static/*` : Fichiers statiques (HTML/JS/CSS)

---

**Bon développement ! 🚀**
