# Analyse du Projet Agent IA BTP - Devis & Factures

## 📋 Vue d'ensemble

Votre projet est un **agent IA spécialisé dans le BTP** qui analyse, prépare et valide des devis et factures. Il utilise une architecture **LangGraph** avec 4 nœuds et s'intègre avec **Supabase** pour le stockage et le RAG vectoriel.

### Architecture actuelle

```
LangGraph Pipeline:
1. InputNormalizerNode → Détecte l'intention et normalise l'entrée
2. RAGRetrieverNode → Récupère le contexte depuis Supabase
3. BusinessToolsNode → Exécute les outils métier (calculs, validations)
4. LLMSynthesizerNode → Génère la réponse finale (JSON ou texte)
```

### Technologies utilisées

- **Backend**: FastAPI + LangGraph + LangChain
- **Base de données**: Supabase (PostgreSQL + pgvector)
- **LLM**: OpenAI GPT-4o-mini (avec fallback GPT-4o)
- **Embeddings**: OpenAI (ou Mistral si configuré)
- **Extraction**: PyPDF, python-docx, pytesseract (OCR)

---

## ✅ Corrections effectuées

### 1. **Modèles OpenAI corrigés** ✅
- ❌ Avant: `gpt-5-mini` et `gpt-4.1` (n'existent pas)
- ✅ Après: `gpt-4o-mini` et `gpt-4o`

### 2. **Fonction dupliquée supprimée** ✅
- Suppression de la première définition de `_format_ai_reply` dans `api.py`
- Conservation de la version avec puces ASCII (meilleure compatibilité)

### 3. **Bug `missing_fields` corrigé** ✅
- La variable `missing_fields` n'était pas définie dans le scope de la fonction
- Ajout de l'extraction depuis `reply` avant utilisation

### 4. **Documentation RAG améliorée** ✅
- Ajout de commentaires expliquant la logique de seuil de similarité

---

## 🔍 Points d'attention identifiés

### 1. **Gestion d'erreurs**
- Beaucoup de `try/except` avec `pass` silencieux
- **Recommandation**: Ajouter un logging structuré pour tracer les erreurs

### 2. **Validation des entrées**
- Certaines validations sont faites dans les outils mais pas au niveau API
- **Recommandation**: Ajouter des validations Pydantic plus strictes dans les endpoints

### 3. **Performance**
- Le RAG récupère toujours k=5 documents même si pas nécessaire
- **Recommandation**: Conditionner le RAG selon l'intention (pas besoin pour chat simple)

### 4. **Tests**
- Aucun test unitaire ou d'intégration visible
- **Recommandation**: Ajouter des tests pour les outils métier critiques

### 5. **Configuration**
- Variables d'environnement non documentées
- **Recommandation**: Créer un fichier `.env.example`

### 6. **Frontend**
- Le frontend React (`nextmind-bid-builder-main`) semble séparé
- **Recommandation**: Vérifier l'intégration avec l'API FastAPI

---

## 🚀 Recommandations d'amélioration

### Priorité Haute

#### 1. **Ajouter un système de logging**
```python
# agent/logging_config.py
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
```

#### 2. **Créer un fichier `.env.example`**
```env
# OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_FALLBACK_MODEL=gpt-4o

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_ANON_KEY=eyJ...
SUPABASE_VECTOR_TABLE=documents
SUPABASE_VECTOR_QUERY_NAME=match_documents

# Optionnel
MISTRAL_API_KEY=...
LANGCHAIN_TRACING_V2=false
AI_CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173
```

#### 3. **Améliorer la gestion d'erreurs**
- Remplacer les `except: pass` par des logs appropriés
- Retourner des erreurs structurées dans les réponses API

### Priorité Moyenne

#### 4. **Optimiser le RAG**
- Ne pas appeler RAG pour les intentions `chat` simples
- Ajouter un cache pour les requêtes fréquentes

#### 5. **Ajouter des tests**
```python
# tests/test_tools.py
def test_clean_lines_tool():
    lines = [{"description": "Test", "quantity": 10, "unit_price_ht": 100}]
    result = clean_lines_tool.invoke({"lines": lines})
    assert len(result["lines"]) == 1
    assert result["lines"][0]["quantity"] == 10.0
```

#### 6. **Améliorer la validation**
- Ajouter des validations Pydantic plus strictes
- Valider les formats de fichiers avant extraction

### Priorité Basse

#### 7. **Documentation API**
- Ajouter OpenAPI/Swagger avec descriptions détaillées
- Documenter les formats de réponse

#### 8. **Monitoring**
- Ajouter des métriques (temps de réponse, taux d'erreur)
- Intégrer avec LangSmith pour le traçage

#### 9. **CI/CD**
- Ajouter GitHub Actions pour les tests
- Automatiser le déploiement Docker

---

## 📊 Structure du projet

```
agent IA/
├── agent/
│   ├── api.py              # Endpoints FastAPI
│   ├── runtime.py          # Graph LangGraph
│   ├── config.py           # Configuration LLM/embeddings
│   ├── tools.py            # Outils métier (calculs, validations)
│   ├── rag.py              # RAG Supabase
│   └── supabase_client.py  # Client Supabase
├── prompts/                # Prompts modulaires
├── templates/              # Templates Jinja2 (devis/factures)
├── static/                 # Frontend HTML/JS
└── supabase/               # Schéma SQL
```

---

## 🎯 Prochaines étapes suggérées

1. **Tester les corrections** : Vérifier que les modèles fonctionnent correctement
2. **Ajouter le logging** : Implémenter un système de logs structuré
3. **Créer `.env.example`** : Documenter les variables d'environnement
4. **Tests unitaires** : Commencer par tester les outils métier
5. **Optimisation RAG** : Conditionner l'appel RAG selon l'intention

---

## 📝 Notes techniques

### Flux de traitement

1. **Analyse de fichier** (`/analyze`):
   - Upload → Extraction texte → Normalisation → RAG → Validation → Upsert Supabase

2. **Préparation devis** (`/prepare-devis`):
   - Formulaire → Nettoyage lignes → Calcul totaux → Synthèse LLM → Validation

3. **Chat** (`/chat`):
   - Message → Normalisation → RAG (si nécessaire) → Réponse concise

### Schéma Supabase

- `clients` : Informations clients
- `devis` / `factures` : Documents principaux
- `devis_items` / `facture_items` : Lignes de détail
- `documents` : Table vectorielle pour RAG
- `embeddings` : Embeddings stockés

---

## 🔗 Ressources utiles

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Supabase Vector Store](https://python.langchain.com/docs/integrations/vectorstores/supabase)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Document généré automatiquement après analyse du code*
