# 💪 HealthMate - Application de Santé Personnelle

HealthMate est une application Streamlit complète qui vous aide à gérer votre santé et votre fitness avec plusieurs outils pratiques.

## 🚀 Fonctionnalités

### 📊 Calculateur d'IMC
- Calcul automatique de l'Indice de Masse Corporelle
- Interprétation des résultats selon les standards médicaux
- Interface intuitive avec métriques visuelles

### 🔥 Calculateur de Calories
- Estimation des besoins caloriques avec la formule Harris-Benedict révisée (Mifflin-St Jeor)
- Calcul du métabolisme de base (BMR) et des besoins totaux (TDEE)
- Prise en compte du niveau d'activité physique

### 💪 Générateur d'Exercices
- Base de données de 24 exercices variés
- Plans d'entraînement personnalisés selon :
  - Durée (15, 30, 45, 60 minutes)
  - Niveau de difficulté (Débutant, Intermédiaire, Avancé)
  - Groupes musculaires ciblés
- Instructions détaillées pour chaque exercice

### 🤖 Chatbot Santé
- Réponses intelligentes aux questions de santé courantes
- Conseils sur l'IMC, les calories, l'exercice, l'hydratation, le sommeil, le stress et l'alimentation
- Interface de chat conviviale avec historique

## 📁 Structure du Projet

```
HealthMate App/
├── app.py                 # Application principale Streamlit
├── requirements.txt       # Dépendances Python
├── data/
│   └── exercises.csv     # Base de données des exercices
└── README.md             # Documentation
```

## 🛠️ Installation et Utilisation

### Prérequis
- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation

1. **Cloner ou télécharger le projet**
   ```bash
   cd "HealthMate App"
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application**
   ```bash
   streamlit run app.py
   ```

4. **Ouvrir dans le navigateur**
   L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 📋 Dépendances

- **streamlit** (≥1.28.0) : Framework web pour l'interface utilisateur
- **pandas** (≥2.0.0) : Manipulation des données et lecture CSV
- **numpy** (≥1.24.0) : Calculs numériques

## 🎯 Utilisation des Fonctionnalités

### Calculateur d'IMC
1. Sélectionnez "📊 Calculateur IMC" dans le menu
2. Entrez votre poids (kg) et taille (m)
3. Cliquez sur "Calculer l'IMC"
4. Consultez votre résultat et son interprétation

### Calculateur de Calories
1. Sélectionnez "🔥 Calculateur Calories" dans le menu
2. Remplissez vos informations personnelles
3. Choisissez votre niveau d'activité
4. Obtenez vos besoins caloriques quotidiens

### Générateur d'Exercices
1. Sélectionnez "💪 Générateur d'Exercices" dans le menu
2. Choisissez la durée et le niveau de difficulté
3. Sélectionnez les groupes musculaires à travailler
4. Générez votre plan d'entraînement personnalisé

### Chatbot Santé
1. Sélectionnez "🤖 Chatbot Santé" dans le menu
2. Posez vos questions de santé
3. Utilisez les suggestions ou tapez vos propres questions
4. Consultez l'historique de la conversation

## 🔧 Personnalisation

### Ajouter des Exercices
Modifiez le fichier `data/exercises.csv` pour ajouter de nouveaux exercices :
- **name** : Nom de l'exercice
- **muscle_group** : Groupe musculaire ciblé
- **difficulty** : Niveau (Débutant, Intermédiaire, Avancé)
- **instructions** : Instructions détaillées
- **duration_reps** : Durée ou nombre de répétitions

### Modifier le Chatbot
Étendez les réponses du chatbot en modifiant le dictionnaire `responses` dans la fonction `get_health_response()` du fichier `app.py`.

## 🎨 Interface Utilisateur

L'application utilise :
- **Design responsive** adapté à tous les écrans
- **Navigation intuitive** avec menu latéral
- **Métriques visuelles** pour les résultats
- **Cartes d'exercices** avec instructions claires
- **Interface de chat** moderne et conviviale

## 📊 Formules Utilisées

### IMC
```
IMC = Poids (kg) / Taille² (m)
```

### Harris-Benedict Révisé (Mifflin-St Jeor)
```
Homme : BMR = 10 × Poids + 6.25 × Taille - 5 × Âge + 5
Femme : BMR = 10 × Poids + 6.25 × Taille - 5 × Âge - 161

TDEE = BMR × Facteur d'activité
```

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créez une branche pour votre fonctionnalité
3. Commitez vos changements
4. Ouvrez une Pull Request



## 🆘 Support

Pour toute question ou problème :
- Ouvrez une issue sur GitHub
- Consultez la documentation Streamlit
- Vérifiez que toutes les dépendances sont installées

---

**HealthMate** - Votre compagnon santé personnel ! 💪
