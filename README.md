# 💪 HealthMate Pro

**Votre compagnon santé intelligent avec IA et RAG**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0.0+-green)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)](https://openai.com/)

## 🎯 Vue d'ensemble

HealthMate Pro est une application web complète de suivi de santé et nutrition, intégrant des fonctionnalités d'intelligence artificielle avancées pour des conseils personnalisés et une analyse nutritionnelle intelligente.

### ✨ Fonctionnalités Principales

#### 🏠 **Interface Moderne**
- Design responsive et intuitif
- Navigation fluide entre les fonctionnalités
- Interface utilisateur professionnelle

#### 📊 **Calculateurs de Santé**
- **Calculateur IMC** : Indice de Masse Corporelle avec interprétation
- **Calculateur de Calories** : Besoins caloriques avec formule Harris-Benedict
- **Calculateur de Macronutriments** : Répartition optimale des nutriments

#### 💪 **Générateur d'Exercices**
- Base de données de 25+ exercices
- Plans personnalisés selon durée et difficulté
- Filtrage par groupes musculaires
- Instructions détaillées pour chaque exercice

#### 🤖 **Chatbot Santé**
- Réponses intelligentes aux questions de santé courantes
- Conseils sur l'IMC, les calories, l'exercice, l'hydratation, le sommeil, le stress et l'alimentation
- Interface de chat conviviale avec historique

#### 🧠 **Analyse Nutritionnelle IA**
- **Analyse d'images de repas** avec reconnaissance automatique d'aliments
- **Analyse de descriptions textuelles** de repas avec conseils personnalisés
- **Base de données nutritionnelle étendue** avec 100+ aliments
- **Score de santé** automatique (0-10) pour chaque repas
- **Recommandations IA** basées sur OpenAI GPT-4o-mini
- **Recherche d'informations nutritionnelles** détaillées
- **Graphiques nutritionnels** interactifs

#### 🧠 **Chatbot RAG Nutrition**
- **Système RAG complet** (Retrieval-Augmented Generation)
- **Upload de documents PDF** nutritionnels personnalisés
- **Extraction automatique de texte** avec PyMuPDF
- **Base vectorielle FAISS** pour la recherche sémantique
- **Modèle local Flan-T5** pour les réponses intelligentes
- **Chat interactif** basé sur vos documents
- **Gestion de session** avec historique des conversations

#### 📈 **Suivi des Progrès**
- Journal quotidien de santé
- Graphiques d'évolution du poids, sommeil et énergie
- Statistiques mensuelles détaillées
- Métriques de progression

#### 🥗 **Nutrition Équilibrée**
- Calculateur de macronutriments personnalisé
- Journal alimentaire quotidien
- Analyse nutritionnelle avec graphiques
- Suivi des calories et nutriments

#### 💧 **Santé Globale**
- **Hydratation** : Suivi de la consommation d'eau avec objectifs
- **Sommeil** : Enregistrement des habitudes de sommeil
- **Bien-être** : Techniques de relaxation et gestion du stress
- Quiz de bien-être avec score personnalisé

#### 🎯 **Objectifs Personnels**
- Définition d'objectifs personnalisés (poids, hydratation, sommeil, exercice)
- Suivi de progression vers les objectifs
- Graphiques de motivation
- Conseils personnalisés

#### 📊 **Dashboard Principal**
- Vue d'ensemble de la santé
- Métriques clés en temps réel
- Graphiques de synthèse
- Recommandations personnalisées

## 📁 Structure du Projet

```
HealthMate App/
├── 📄 app.py                          # Application principale Streamlit
├── 📄 requirements.txt                # Dépendances Python
├── 📄 README.md                       # Documentation
├── 📄 start.py                        # Script de démarrage rapide
├── 📄 test_app.py                     # Tests de l'application
├── 📄 test_rag.py                     # Tests du système RAG
├── 📄 test_rag_simple.py              # Tests RAG simplifiés
├── 📁 data/                           # Données de l'application
│   ├── 📄 exercises.csv               # Base de données des exercices
│   ├── 📄 nutrition_database.csv      # Base de données nutritionnelle
│   ├── 📄 guide_nutrition_complet.pdf # Guide nutrition complet
│   └── 📄 gestion_poids.pdf           # Guide gestion du poids
└── 📁 rag_demo_index/                 # Index RAG (généré automatiquement)
    ├── 📄 index.faiss                 # Index vectoriel FAISS
    └── 📄 index.pkl                   # Métadonnées de l'index
```

## 🚀 Installation et Démarrage

### Option 1: Démarrage Rapide (Recommandé)

```bash
# Cloner ou télécharger le projet
cd "HealthMate App"

# Démarrage automatique (installe les dépendances et démarre l'app)
python start.py
```

### Option 2: Installation Manuelle

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Démarrer l'application
streamlit run app.py
```


## 📋 Dépendances

### Dépendances de Base
- **streamlit** (≥1.28.0) : Framework web pour l'interface utilisateur
- **pandas** (≥2.0.0) : Manipulation des données et lecture CSV
- **numpy** (≥1.24.0) : Calculs numériques
- **plotly** (≥5.15.0) : Graphiques interactifs

### Dépendances pour l'Analyse Nutritionnelle IA
- **openai** (≥1.0.0) : API OpenAI pour l'analyse IA
- **langchain** (≥0.1.0) : Framework pour les applications LLM
- **langchain-core** (≥0.1.0) : Composants de base LangChain
- **langchain-openai** (≥0.1.0) : Intégration OpenAI avec LangChain
- **pillow** (≥10.0.0) : Traitement d'images
- **torch** (≥2.0.0) : Framework de deep learning
- **torchvision** (≥0.15.0) : Vision par ordinateur avec PyTorch
- **timm** (≥0.9.0) : Modèles pré-entraînés pour la vision

### Dépendances pour le Chatbot RAG
- **langchain-community** (≥0.1.0) : Composants communautaires LangChain
- **langchain-text-splitters** (≥0.1.0) : Division de texte pour RAG
- **faiss-cpu** (≥1.7.0) : Base de données vectorielle pour la recherche
- **sentence-transformers** (≥2.2.0) : Modèles d'embeddings sémantiques
- **transformers** (≥4.30.0) : Modèles de transformation de texte
- **pymupdf** (≥1.23.0) : Extraction de texte depuis les PDFs
- **accelerate** (≥0.20.0) : Optimisation des modèles
- **bitsandbytes** (≥0.41.0) : Quantification des modèles
- **tqdm** (≥4.65.0) : Barres de progression

### Dépendances Optionnelles
- **reportlab** (≥4.0.0) : Génération de PDFs (pour les tests)

## 🎯 Utilisation des Fonctionnalités

### Calculateur d'IMC
1. Sélectionnez "📊 Calculateur IMC" dans le menu
2. Entrez votre poids et taille
3. Cliquez sur "Calculer l'IMC"
4. Consultez votre résultat et l'interprétation

### Calculateur de Calories
1. Sélectionnez "🔥 Calculateur Calories" dans le menu
2. Remplissez vos informations personnelles
3. Sélectionnez votre niveau d'activité
4. Obtenez votre BMR et TDEE

### Générateur d'Exercices
1. Sélectionnez "💪 Générateur d'Exercices" dans le menu
2. Choisissez la durée et la difficulté
3. Sélectionnez les groupes musculaires
4. Cliquez sur "Générer le Plan"

### Chatbot Santé
1. Sélectionnez "🤖 Chatbot Santé" dans le menu
2. Posez vos questions de santé
3. Utilisez les suggestions ou tapez vos propres questions
4. Consultez l'historique de la conversation

### 🧠 Analyse Nutritionnelle IA
1. **Configuration** : Configurez votre clé API OpenAI dans l'onglet dédié
2. **Analyse d'Image** :
   - Téléchargez une photo de votre repas
   - L'IA reconnaît automatiquement les aliments
   - Obtenez un score de santé et des recommandations
3. **Analyse de Texte** :
   - Décrivez votre repas en texte
   - Recevez une analyse nutritionnelle détaillée
   - Consultez les suggestions d'amélioration
4. **Recherche d'Aliments** :
   - Recherchez des informations nutritionnelles détaillées
   - Consultez les graphiques nutritionnels
   - Explorez la base de données de 100+ aliments

### 🧠 Chatbot RAG Nutrition
1. **Upload de Documents** :
   - Téléchargez des PDFs de documents nutritionnels
   - Le système extrait automatiquement le texte
   - Construisez votre base de connaissances personnalisée
2. **Chat Intelligent** :
   - Posez des questions sur la nutrition
   - Recevez des réponses basées sur vos documents
   - Utilisez les suggestions de questions
3. **Gestion du Système** :
   - Consultez le statut du système RAG
   - Rechargez ou réinitialisez selon vos besoins
   - Suivez les statistiques d'utilisation

### Suivi des Progrès
1. Sélectionnez "📈 Suivi des Progrès" dans le menu
2. Enregistrez vos données quotidiennes
3. Consultez les graphiques d'évolution
4. Analysez vos statistiques mensuelles

### Nutrition Équilibrée
1. Sélectionnez "🥗 Nutrition Équilibrée" dans le menu
2. Calculez vos besoins en macronutriments
3. Enregistrez vos repas dans le journal alimentaire
4. Consultez l'analyse nutritionnelle

### Santé Globale
1. Sélectionnez "💧 Santé Globale" dans le menu
2. Suivez votre hydratation quotidienne
3. Enregistrez vos habitudes de sommeil
4. Pratiquez les techniques de relaxation

### Mes Objectifs
1. Sélectionnez "🎯 Mes Objectifs" dans le menu
2. Définissez vos objectifs personnels
3. Suivez votre progression
4. Consultez les conseils personnalisés

### Dashboard
1. Sélectionnez "📊 Dashboard" dans le menu
2. Consultez votre vue d'ensemble
3. Analysez les métriques clés
4. Suivez les recommandations

## 🔧 Personnalisation

### Ajouter des Exercices
Modifiez le fichier `data/exercises.csv` pour ajouter de nouveaux exercices :

```csv
name,muscle_group,difficulty,instructions,duration_reps
Nouvel Exercice,Groupe Musculaire,Difficulté,Instructions détaillées,Durée/Répétitions
```

### Étendre la Base Nutritionnelle
Ajoutez des aliments au fichier `data/nutrition_database.csv` :

```csv
food,calories,protein,fat,fiber,carbohydrates,sodium,calcium,iron,vitamin_c
Nouvel Aliment,Calories,Protéines,Lipides,Fibres,Glucides,Sodium,Calcium,Fer,Vitamine C
```

### Configuration de l'Analyse IA
1. Obtenez une clé API OpenAI sur [platform.openai.com](https://platform.openai.com)
2. Configurez la clé dans l'application
3. Utilisez les fonctionnalités d'analyse IA

### Configuration du RAG
1. Téléchargez vos documents PDF nutritionnels
2. Le système construit automatiquement l'index
3. Posez vos questions sur vos documents

## 🧪 Tests

### Tests de l'Application
```bash
python test_app.py
```

### Tests du Système RAG
```bash
python test_rag.py
python test_rag_simple.py
```

### Génération de PDFs de Test
```bash
python create_nutrition_pdfs.py
```


## 🚀 Fonctionnalités Avancées

### Intégration IA
- **OpenAI GPT-4o-mini** : Analyse nutritionnelle intelligente
- **Modèles de vision** : Reconnaissance d'aliments dans les images
- **LangChain** : Orchestration des modèles IA

### Système RAG
- **FAISS** : Recherche vectorielle rapide
- **Sentence Transformers** : Embeddings sémantiques
- **Flan-T5** : Modèle de génération de texte local
- **PyMuPDF** : Extraction de texte depuis PDFs

### Interface Moderne
- **Streamlit** : Framework web moderne
- **Plotly** : Graphiques interactifs
- **CSS personnalisé** : Design professionnel

## 📈 Roadmap

### Version Actuelle (1.0.0)
- ✅ Interface utilisateur complète
- ✅ Calculateurs de santé
- ✅ Système d'exercices
- ✅ Chatbot santé
- ✅ Analyse nutritionnelle IA
- ✅ Chatbot RAG
- ✅ Suivi des progrès
- ✅ Dashboard complet



## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commit vos changements
4. Push vers la branche
5. Ouvrir une Pull Request



**💪 HealthMate Pro - Transformez Votre Santé, Transformez Votre Vie**
