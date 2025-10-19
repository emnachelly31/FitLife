#!/usr/bin/env python3
"""
Script de démarrage rapide pour HealthMate App
"""

import os
import sys
import subprocess

def check_dependencies():
    """Vérifie si les dépendances sont installées"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        print("✅ Dépendances de base installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendances manquantes: {e}")
        return False

def install_dependencies():
    """Installe les dépendances"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False

def create_sample_data():
    """Crée les données d'exemple si elles n'existent pas"""
    if not os.path.exists("data"):
        os.makedirs("data")
        print("📁 Dossier 'data' créé")
    
    if not os.path.exists("data/exercises.csv"):
        print("📄 Création du fichier exercises.csv...")
        # Créer un fichier d'exercices basique
        exercises_data = """name,muscle_group,difficulty,instructions,duration_reps
Pompes,Poitrine,Débutant,Position planche,3x10
Squats,Jambes,Débutant,Pieds écartés largeur épaules,3x15
Planche,Abdominaux,Débutant,Position statique,3x30s
Burpees,Cardio,Intermédiaire,Exercice complet,3x8
Développé couché,Poitrine,Intermédiaire,Avec haltères,3x8
Tractions,Dos,Avancé,Barre de traction,3x5"""
        
        with open("data/exercises.csv", "w", encoding="utf-8") as f:
            f.write(exercises_data)
        print("✅ Fichier exercises.csv créé")
    
    if not os.path.exists("data/nutrition_database.csv"):
        print("📄 Création du fichier nutrition_database.csv...")
        # Créer un fichier de base nutritionnelle basique
        nutrition_data = """food,calories,protein,fat,fiber,carbohydrates,sodium,calcium,iron,vitamin_c
Pomme,52,0.3,0.2,2.4,14,1,6,0.1,4.6
Banane,89,1.1,0.3,2.6,23,1,5,0.3,8.7
Poulet,165,31,3.6,0,0,74,15,0.9,0
Saumon,208,25,12,0,0,59,12,0.8,0
Brocoli,34,2.8,0.4,2.6,7,33,47,0.7,89"""
        
        with open("data/nutrition_database.csv", "w", encoding="utf-8") as f:
            f.write(nutrition_data)
        print("✅ Fichier nutrition_database.csv créé")

def start_app():
    """Démarre l'application Streamlit"""
    print("🚀 Démarrage de HealthMate App...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Application fermée")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")

def main():
    """Fonction principale"""
    print("💪 HealthMate App - Démarrage Rapide")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("\n🔧 Installation des dépendances...")
        if not install_dependencies():
            print("❌ Impossible d'installer les dépendances")
            return
    
    # Créer les données d'exemple
    print("\n📊 Préparation des données...")
    create_sample_data()
    
    # Démarrer l'application
    print("\n🎯 Tout est prêt !")
    start_app()

if __name__ == "__main__":
    main()
