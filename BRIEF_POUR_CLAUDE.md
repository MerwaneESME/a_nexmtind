# 📋 Brief Projet : Agent IA BTP - Devis & Factures

## 🎯 Contexte et Objectif

Ce projet est un **agent IA spécialisé dans le secteur BTP (Bâtiment et Travaux Publics)** qui automatise la création, l'analyse et la validation de devis et factures. L'objectif est d'aider les professionnels du BTP à :
- Analyser automatiquement des documents existants (PDF, DOCX, images)
- Générer des devis/factures conformes aux réglementations françaises
- Détecter les incohérences (totaux, TVA, mentions légales)
- Pré-remplir les documents depuis un historique de clients et matériaux

**Domaine métier** : BTP français (réglementation TVA, pénalités de retard, RC pro, mentions légales obligatoires)

---

## 🏗️ Architecture Technique

### Stack Technologique

- **Backend** : FastAPI (Python 3.11+)
- **Orchestration IA** : LangGraph (workflow stateful)
- **LLM Framework** : LangChain
- **Modèles LLM** : OpenAI GPT-4o-mini (principal) + GPT-4o (fallback)
- **Embeddings** : OpenAI (par défaut) ou Mistral (optionnel)
- **Base de données** : Supabase (PostgreSQL + pgvector pour RAG)
- **Extraction documents** : PyPDF, python-docx, pytesseract (OCR)
- **Templates** : Jinja2 pour génération de documents

### Architecture LangGraph (4 nœuds séquentiels)

```
┌─────────────────────┐
│ InputNormalizerNode │ → Détecte l'intention et normalise l'entrée
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ RAGRetrieverNode    │ → Récupère contexte depuis Supabase (clients, matériaux, historiques)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ BusinessToolsNode   │ → Exécute outils métier (calculs, validations, extraction)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ LLMSynthesizerNode  │ → Génère réponse finale (JSON structuré ou texte)
└─────────────────────┘
```

**Caractéristiques** :
- Workflow séquentiel (pas de branchements conditionnels pour l'instant)
- Mémoire par thread (MemorySaver de LangGraph)
- State typé avec TypedDict (AgentState)

---

## 📁 Structure du Code

```
agent IA/
├── agent/
│   ├── api.py              # Endpoints FastAPI (chat, analyze, prepare-devis)
│   ├── runtime.py          # Graph LangGraph + nœuds (4 fonctions principales)
│   ├── config.py           # Configuration LLM, embeddings, schémas Pydantic
│   ├── tools.py            # Outils métier (extraction, calculs, validation)
│   ├── rag.py              # Wrapper SupabaseVectorStore pour RAG
│   ├── supabase_client.py   # Client Supabase + fonctions upsert
│   ├── supabase_helpers.py # Helpers Supabase (clients récents, matériaux)
│   ├── create_devis.py     # Endpoint alternatif création devis
│   ├── dashboard.py        # Fonctions dashboard (agrégats, statistiques)
│   ├── search.py           # Recherche documents Supabase
│   ├── suggestions.py      # Suggestions intelligentes
│   └── alerts.py           # Système d'alertes
├── prompts/                # Prompts modulaires (Jinja2)
│   ├── analysis_prompt.txt      # Normalisation intention
│   ├── chat_prompt.txt          # Chat conversationnel
│   ├── prepare_devis_prompt.txt # Synthèse finale JSON
│   ├── rag_prompt.txt           # Génération requêtes RAG
│   └── validate_prompt.txt      # Validation conformité
├── templates/              # Templates Jinja2
│   ├── quote.docx.j2      # Template devis
│   └── invoice.docx.j2    # Template facture
├── static/                # Frontend HTML/JS/CSS simple
├── supabase/              # Schéma SQL
│   └── schema.sql
└── requirements.txt       # Dépendances Python
```

---

## 🔄 Flux de Données Principaux

### 1. Analyse de fichier (`POST /analyze`)

```
Fichier uploadé (PDF/DOCX/IMG)
    ↓
extract_pdf_tool → Extraction texte
    ↓
InputNormalizerNode → Détection intention "analyze"
    ↓
RAGRetrieverNode → Recherche contexte similaire (k=5, seuil 0.75)
    ↓
BusinessToolsNode → 
    - clean_lines_tool (nettoyage quantités/prix)
    - calculate_totals_tool (HT/TVA/TTC)
    - validate_devis_tool (conformité)
    - supabase_lookup_tool (pré-remplissage client)
    ↓
LLMSynthesizerNode → JSON structuré avec corrections
    ↓
upsert_document → Sauvegarde Supabase
    ↓
Réponse API : {data, formatted, totals, corrections, errors}
```

### 2. Préparation devis (`POST /prepare-devis`)

```
Formulaire (client, lignes, notes)
    ↓
InputNormalizerNode → Intent "prepare_devis"
    ↓
BusinessToolsNode → Nettoyage + calculs + validation
    ↓
LLMSynthesizerNode → Synthèse JSON strict
    ↓
Validation + Upsert Supabase
    ↓
Réponse : {data, formatted, totals, corrections}
```

### 3. Chat conversationnel (`POST /chat`)

```
Message utilisateur
    ↓
InputNormalizerNode → Intent "chat" (ou "prepare_devis" si détecté)
    ↓
RAGRetrieverNode → Contexte si nécessaire
    ↓
BusinessToolsNode → Outils si besoin (calculs, lookup)
    ↓
LLMSynthesizerNode → Réponse concise JSON {reply, todo}
    ↓
Réponse formatée pour frontend
```

---

## 🛠️ Outils Métier Détailés

### `extract_pdf_tool`
- **Entrée** : Chemin fichier (PDF/DOCX/PNG/JPG)
- **Sortie** : Texte extrait + métadonnées
- **Technologies** : PyPDF, python-docx, pytesseract

### `clean_lines_tool`
- **Fonction** : Normalise les lignes de devis/facture
- **Actions** :
  - Normalise quantités, prix, TVA, remises
  - Détecte doublons (par description)
  - Corrige valeurs négatives
  - Valide formats

### `calculate_totals_tool`
- **Fonction** : Calcule HT/TVA/TTC
- **Détecte** : Incohérences numériques, lignes à zéro, TVA manquante

### `validate_devis_tool`
- **Contrôles** :
  - Cohérence totaux (écarts > 0.01€)
  - Mentions obligatoires (conditions paiement, pénalités retard, RC pro)
  - TVA par ligne (détecte TVA à 0% ou absente)
  - Dates échéance (factures)
- **Sortie** : Liste d'issues avec sévérité (high/medium/low)

### `supabase_lookup_tool`
- **Modes** : clients, materials, history, prefill, auto
- **Fonction** : Recherche dans Supabase pour pré-remplissage
- **Tables** : `clients`, `devis_items`, `devis`, `factures`

---

## 🗄️ Schéma Supabase

### Tables principales

- **`clients`** : Informations clients (id, name, address, contact JSONB)
- **`devis`** : Devis (id, user_id, client_id, status, metadata JSONB, total)
- **`devis_items`** : Lignes de devis (devis_id, description, qty, unit_price, total)
- **`factures`** : Factures (id, user_id, client_id, metadata JSONB, total, devis_id)
- **`facture_items`** : Lignes de facture
- **`documents`** : Table vectorielle pour RAG (embedding vector, metadata)
- **`embeddings`** : Embeddings stockés (source_table, source_id, embedding vector)

### Relations
- `devis.client_id` → `clients.id`
- `factures.client_id` → `clients.id`
- `factures.devis_id` → `devis.id` (facture liée à un devis)

---

## 🎨 Prompts et Stratégie

### Philosophie des prompts
- **Concis** : Pas de prose, JSON strict uniquement
- **Actionnable** : Réponses orientées correction/action
- **Modulaires** : Un prompt par nœud LangGraph
- **Métier BTP** : Connaissance réglementaire française intégrée

### Exemples de prompts

**analysis_prompt** : Détecte intention + normalise payload
**chat_prompt** : Réponses courtes avec checklist (max 4 items)
**prepare_devis_prompt** : Synthèse JSON strict avec corrections
**rag_prompt** : Génère 1-3 requêtes ciblées pour recherche vectorielle
**validate_prompt** : Liste issues + suggestions corrections

---

## 📊 Points Forts Actuels

1. **Architecture modulaire** : Séparation claire des responsabilités
2. **RAG intégré** : Contexte historique pour pré-remplissage intelligent
3. **Validation métier** : Détection automatique d'incohérences
4. **Multi-format** : Support PDF/DOCX/images avec OCR
5. **Conformité réglementaire** : Mentions légales BTP français
6. **API RESTful** : Endpoints clairs et documentés

---

## ⚠️ Points d'Attention / Limitations

### Techniques
1. **Gestion d'erreurs** : Beaucoup de `try/except: pass` silencieux
2. **Logging** : Pas de système de logs structuré
3. **Tests** : Aucun test unitaire ou d'intégration
4. **Performance RAG** : Toujours appelé même pour chats simples
5. **Validation API** : Validations Pydantic incomplètes
6. **Gestion fichiers** : Fichiers temporaires pas toujours nettoyés

### Fonctionnelles
1. **Workflow linéaire** : Pas de branchements conditionnels dans LangGraph
2. **Mémoire** : Mémoire par thread mais pas de persistance long terme
3. **Multi-utilisateurs** : Pas de gestion d'authentification explicite
4. **Templates** : Génération PDF limitée (HTML → PDF via WeasyPrint/Playwright)

---

## 🔍 Questions pour Claude

### Architecture & Design
1. **Comment améliorer le workflow LangGraph ?** 
   - Ajouter des branchements conditionnels selon l'intention ?
   - Optimiser l'appel RAG (ne pas l'appeler pour chats simples) ?
   - Implémenter un système de retry/fallback plus robuste ?

2. **Stratégie de mémoire et contexte**
   - Comment gérer la mémoire long terme (historique conversations) ?
   - Quelle stratégie pour limiter le contexte LLM (RAG vs mémoire conversationnelle) ?
   - Comment gérer les sessions multi-utilisateurs efficacement ?

### Performance & Scalabilité
3. **Optimisations possibles**
   - Cache pour requêtes RAG fréquentes ?
   - Traitement asynchrone pour extraction fichiers lourds ?
   - Batch processing pour plusieurs documents ?

4. **Monitoring & Observabilité**
   - Quelle stratégie de logging structuré (structlog, loguru) ?
   - Intégration LangSmith pour traçage complet ?
   - Métriques à suivre (temps réponse, taux erreur, coût tokens) ?

### Qualité & Robustesse
5. **Tests et validation**
   - Structure de tests recommandée (unitaires, intégration, e2e) ?
   - Comment tester les nœuds LangGraph individuellement ?
   - Tests de non-régression pour prompts ?

6. **Gestion d'erreurs**
   - Stratégie de retry pour appels LLM/API ?
   - Gestion gracieuse des échecs (fallback, messages utilisateur) ?
   - Validation entrées plus stricte (Pydantic models) ?

### Fonctionnalités Métier
7. **Améliorations BTP**
   - Intégration référentiels DTU (Documents Techniques Unifiés) ?
   - Calcul automatique quantités depuis plans/descriptions ?
   - Suggestions prix matériaux depuis bases de données externes ?

8. **Génération documents**
   - Meilleure génération PDF (templates Word → PDF) ?
   - Support signatures électroniques ?
   - Export formats multiples (PDF, Excel, XML) ?

### Intégration & Déploiement
9. **CI/CD et DevOps**
   - Pipeline CI/CD recommandé (tests, lint, déploiement) ?
   - Stratégie déploiement (Docker, Kubernetes, serverless) ?
   - Gestion secrets (variables d'environnement, Vault) ?

10. **Frontend**
    - Intégration avec React/Next.js existant ?
    - WebSockets pour mises à jour temps réel ?
    - Interface admin pour monitoring ?

---

## 📈 Métriques de Succès Potentielles

- **Précision extraction** : % de champs correctement extraits depuis PDF
- **Taux de conformité** : % de devis/factures sans erreurs critiques
- **Temps de génération** : Temps moyen pour créer un devis
- **Taux d'utilisation RAG** : % de requêtes bénéficiant du contexte historique
- **Coût par document** : Coût tokens OpenAI par devis/facture traité
- **Satisfaction utilisateur** : Feedback sur qualité des corrections suggérées

---

## 🎯 Objectifs à Court Terme

1. ✅ Corrections bugs critiques (modèles OpenAI, fonctions dupliquées)
2. 🔄 Ajout système de logging structuré
3. 🔄 Tests unitaires outils métier
4. 🔄 Optimisation RAG (conditionnel selon intention)
5. 🔄 Documentation API complète

---

## 📚 Contexte Technique Supplémentaire

### Variables d'environnement clés
- `OPENAI_API_KEY` : Clé API OpenAI (obligatoire)
- `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` : Configuration Supabase
- `LLM_MODEL` : Modèle principal (défaut: gpt-4o-mini)
- `LLM_FALLBACK_MODEL` : Modèle fallback (défaut: gpt-4o)
- `AI_CORS_ALLOW_ORIGINS` : Origines CORS autorisées

### Dépendances critiques
- `langchain`, `langgraph` : Framework orchestration
- `langchain-openai` : Intégration OpenAI
- `langchain-community` : SupabaseVectorStore
- `fastapi`, `uvicorn` : API REST
- `supabase` : Client Supabase
- `pypdf`, `python-docx` : Extraction documents

---

## 💡 Contexte Métier BTP Français

### Réglementations importantes
- **TVA** : 20% standard, 10% rénovation, 5.5% travaux énergétiques
- **Pénalités retard** : 3x le taux d'intérêt légal + 40€ forfait
- **RC Pro** : Mention obligatoire sur factures
- **Mentions légales** : SIRET, SIREN, TVA intracommunautaire
- **DTU** : Références normes techniques (optionnel mais recommandé)

### Types de documents
- **Devis** : Proposition commerciale (validité limitée)
- **Facture** : Document comptable (obligatoire après travaux)
- **Acompte** : Facture partielle avant travaux
- **Solde** : Facture finale après réception

---

**Merci Claude pour tes conseils et idées ! 🚀**

*Ce document est une synthèse complète du projet pour faciliter la compréhension et obtenir des recommandations pertinentes.*
