import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import BytesIO
import base64
 

# Configuration de la page
st.set_page_config(
    page_title="HealthMate Pro",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .exercise-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    .bot-message {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        margin-right: 2rem;
    }
    
    /* Style pour les boutons de connexion/inscription */
    .login-signup-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.4rem;
        padding: 0.4rem 0.8rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .login-signup-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour calculer l'IMC
def calculate_bmi(weight, height):
    """Calcule l'IMC et retourne la valeur avec l'interprétation"""
    if height <= 0:
        return None, "Taille invalide"
    
    bmi = weight / (height ** 2)
    
    if bmi < 18.5:
        interpretation = "Insuffisance pondérale"
        color = "blue"
    elif bmi < 25:
        interpretation = "Poids normal"
        color = "green"
    elif bmi < 30:
        interpretation = "Surpoids"
        color = "orange"
    else:
        interpretation = "Obésité"
        color = "red"
    
    return round(bmi, 1), interpretation, color

# Fonction pour calculer les calories avec Harris-Benedict
def calculate_calories(weight, height, age, gender, activity_level):
    """Calcule les besoins caloriques avec la formule Harris-Benedict"""
    
    # Formule Harris-Benedict révisée (Mifflin-St Jeor Equation)
    if gender == "Homme":
        bmr = 10 * weight + 6.25 * height * 100 - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height * 100 - 5 * age - 161
    
    # Facteurs d'activité
    activity_factors = {
        "Sédentaire": 1.2,
        "Légèrement actif": 1.375,
        "Modérément actif": 1.55,
        "Très actif": 1.725,
        "Extrêmement actif": 1.9
    }
    
    tdee = bmr * activity_factors[activity_level]
    
    return int(bmr), int(tdee)

# Fonction pour charger les exercices
def load_exercises():
    """Charge la base de données des exercices"""
    try:
        # Utiliser des paramètres robustes pour le parsing CSV
        df = pd.read_csv('data/exercises.csv',
                        encoding='utf-8',
                        quotechar='"',
                        skipinitialspace=True)
        return df
    except FileNotFoundError:
        st.error("Fichier exercises.csv non trouvé. Veuillez créer le fichier data/exercises.csv")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Erreur de parsing du fichier CSV: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des exercices: {e}")
        return None

# Fonctions pour le suivi des progrès
def initialize_session_state():
    """Initialise les variables de session state"""
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = []
    if 'nutrition_data' not in st.session_state:
        st.session_state.nutrition_data = []
    if 'water_intake' not in st.session_state:
        st.session_state.water_intake = 0
    if 'sleep_data' not in st.session_state:
        st.session_state.sleep_data = []
    if 'goals' not in st.session_state:
        st.session_state.goals = {
            'weight_goal': 0,
            'water_goal': 2500,  # ml
            'sleep_goal': 8,     # heures
            'exercise_goal': 150  # minutes par semaine
        }

def add_progress_entry(weight, sleep_hours, energy_level, mood, notes):
    """Ajoute une entrée de progression"""
    entry = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'weight': weight,
        'sleep_hours': sleep_hours,
        'energy_level': energy_level,
        'mood': mood,
        'notes': notes
    }
    st.session_state.progress_data.append(entry)
    st.success("Données de progression enregistrées !")

def calculate_macros(weight, height, age, gender, activity_level, goal):
    """Calcule les macronutriments recommandés"""
    # Calcul du BMR
    if gender == "Homme":
        bmr = 10 * weight + 6.25 * height * 100 - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height * 100 - 5 * age - 161
    
    # Facteurs d'activité
    activity_factors = {
        "Sédentaire": 1.2,
        "Légèrement actif": 1.375,
        "Modérément actif": 1.55,
        "Très actif": 1.725,
        "Extrêmement actif": 1.9
    }
    
    tdee = bmr * activity_factors[activity_level]
    
    # Ajustement selon l'objectif
    if goal == "Perte de poids":
        calories = tdee - 500  # Déficit de 500 kcal
    elif goal == "Prise de muscle":
        calories = tdee + 300  # Surplus de 300 kcal
    else:  # Maintien
        calories = tdee
    
    # Répartition des macronutriments
    proteins = (calories * 0.25) / 4  # 25% des calories, 4 kcal/g
    carbs = (calories * 0.45) / 4     # 45% des calories, 4 kcal/g
    fats = (calories * 0.30) / 9      # 30% des calories, 9 kcal/g
    
    return {
        'calories': round(calories),
        'proteins': round(proteins),
        'carbs': round(carbs),
        'fats': round(fats)
    }

def add_nutrition_entry(meal_type, food, calories, proteins, carbs, fats):
    """Ajoute une entrée nutritionnelle"""
    entry = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'meal_type': meal_type,
        'food': food,
        'calories': calories,
        'proteins': proteins,
        'carbs': carbs,
        'fats': fats
    }
    st.session_state.nutrition_data.append(entry)
    st.success(f"{meal_type} ajouté avec succès !")

def get_daily_nutrition_summary(date=None):
    """Calcule le résumé nutritionnel quotidien"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    daily_entries = [entry for entry in st.session_state.nutrition_data 
                    if entry['date'] == date]
    
    if not daily_entries:
        return {
            'calories': 0,
            'proteins': 0,
            'carbs': 0,
            'fats': 0,
            'meals_count': 0
        }
    
    total_calories = sum(entry['calories'] for entry in daily_entries)
    total_proteins = sum(entry['proteins'] for entry in daily_entries)
    total_carbs = sum(entry['carbs'] for entry in daily_entries)
    total_fats = sum(entry['fats'] for entry in daily_entries)
    
    return {
        'calories': total_calories,
        'proteins': total_proteins,
        'carbs': total_carbs,
        'fats': total_fats,
        'meals_count': len(daily_entries)
    }

def create_progress_chart(data, metric, title):
    """Crée un graphique de progression"""
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    fig = px.line(df, x='date', y=metric, title=title,
                  labels={'date': 'Date', metric: metric.replace('_', ' ').title()})
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified'
    )
    return fig

def create_nutrition_chart(data, days=7):
    """Crée un graphique nutritionnel"""
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrer les derniers jours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = df[df['date'] >= start_date]
    
    # Grouper par date
    daily_summary = df.groupby('date').agg({
        'calories': 'sum',
        'proteins': 'sum',
        'carbs': 'sum',
        'fats': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Calories', 'Protéines (g)', 'Glucides (g)', 'Lipides (g)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=daily_summary['date'], y=daily_summary['calories'], 
                  name='Calories', line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_summary['date'], y=daily_summary['proteins'], 
                  name='Protéines', line=dict(color='#4ECDC4')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=daily_summary['date'], y=daily_summary['carbs'], 
                  name='Glucides', line=dict(color='#45B7D1')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_summary['date'], y=daily_summary['fats'], 
                  name='Lipides', line=dict(color='#96CEB4')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Évolution Nutritionnelle")
    return fig

# Fonction pour générer un plan d'exercices
def generate_exercise_plan(exercises_df, duration, difficulty, muscle_groups):
    """Génère un plan d'exercices personnalisé"""
    if exercises_df is None:
        return []
    
    # Filtrer par difficulté et groupes musculaires
    filtered_exercises = exercises_df[
        (exercises_df['difficulty'] == difficulty) &
        (exercises_df['muscle_group'].isin(muscle_groups))
    ]
    
    if filtered_exercises.empty:
        return []
    
    # Calculer le nombre d'exercices basé sur la durée
    exercises_per_duration = {
        15: 3,
        30: 5,
        45: 7,
        60: 10
    }
    
    num_exercises = exercises_per_duration.get(duration, 5)
    
    # Sélectionner des exercices aléatoires
    selected_exercises = filtered_exercises.sample(n=min(num_exercises, len(filtered_exercises)))
    
    return selected_exercises.to_dict('records')

# Fonction pour le chatbot santé
def get_health_response(user_input):
    """Répond aux questions de santé basiques"""
    user_input = user_input.lower()
    
    responses = {
        "imc": "L'IMC (Indice de Masse Corporelle) est calculé en divisant le poids (kg) par la taille au carré (m). Il aide à évaluer si votre poids est adapté à votre taille.",
        "calories": "Les calories sont l'unité d'énergie des aliments. Vos besoins dépendent de votre âge, sexe, taille, poids et niveau d'activité.",
        "exercice": "L'exercice régulier améliore la santé cardiovasculaire, renforce les muscles et aide à maintenir un poids santé.",
        "eau": "Il est recommandé de boire environ 2-2.5 litres d'eau par jour, mais cela varie selon l'activité physique et le climat.",
        "sommeil": "Un adulte devrait dormir 7-9 heures par nuit pour une santé optimale.",
        "stress": "Le stress peut être géré par la méditation, l'exercice, une alimentation équilibrée et un sommeil suffisant.",
        "alimentation": "Une alimentation équilibrée inclut des fruits, légumes, protéines maigres, céréales complètes et peu de sucre ajouté."
    }
    
    for keyword, response in responses.items():
        if keyword in user_input:
            return response
    
    return "Je suis là pour vous aider avec vos questions de santé. Vous pouvez me demander des informations sur l'IMC, les calories, l'exercice, l'hydratation, le sommeil, le stress ou l'alimentation."

 

# Interface principale
def main():
   
    # Initialisation des données de session
    initialize_session_state()
    
    # Sidebar pour la navigation
    st.sidebar.title("💪 HealthMate Pro")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choisissez une fonctionnalité",
        [
            "🏠 Accueil", 
            "📊 Calculateur IMC", 
            "🔥 Calculateur Calories", 
            "💪 Générateur d'Exercices", 
            "🤖 Chatbot Santé",
            "📈 Suivi des Progrès",
            "🥗 Nutrition Équilibrée",
            "💧 Santé Globale",
            "🎯 Mes Objectifs",
            "📊 Dashboard"
        ]
    )
    
    # Page d'accueil minimisée et centrée
    if page == "🏠 Accueil":
        # Boutons de connexion/inscription en haut à droite
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col2:
            if st.button("🔐 Connexion", key="login_btn"):
                st.info("Fonctionnalité de connexion à venir !")
        
        with col3:
            if st.button("📝 Inscription", key="signup_btn"):
                st.info("Fonctionnalité d'inscription à venir !")
        
        # Contenu centré et minimaliste
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <h1 style="font-size: 3.5rem; color: #667eea; margin-bottom: 0.5rem; font-weight: 700;">
                💪 HealthMate
            </h1>
            <h2 style="font-size: 1.4rem; color: #6c757d; margin-bottom: 2rem; font-weight: 300;">
                "Transformez Votre Santé, Transformez Votre Vie"
            </h2>
            <p style="font-size: 1.1rem; color: #6c757d; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                Votre compagnon santé intelligent avec des outils personnalisés pour votre bien-être.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques centrées
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("IMC Normal", "18.5 - 24.9", "kg/m²")
        
        with col2:
            st.metric("Calories Moyennes", "2000", "kcal/jour")
        
        with col3:
            st.metric("Exercice Recommandé", "150", "min/semaine")
        
        with col4:
            st.metric("Eau Quotidienne", "2.5", "litres")
        
        # Footer
        st.markdown("""
        <div style="background: #2c3e50; color: white; padding: 2rem; border-radius: 0.8rem; 
                    margin-top: 3rem; text-align: center;">
            <h3 style="font-size: 1.5rem; margin-bottom: 0.5rem; font-weight: 600;">
                💪 HealthMate
            </h3>
            <p style="font-size: 1rem; opacity: 0.8; margin-bottom: 1rem;">
                Votre compagnon santé intelligent
            </p>
            <p style="font-size: 0.9rem; opacity: 0.7; margin: 0;">
                © 2025 HealthMate - "Transformez Votre Santé, Transformez Votre Vie"
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Calculateur IMC
    elif page == "📊 Calculateur IMC":
        st.header("📊 Calculateur d'IMC")
        st.markdown("Calculez votre Indice de Masse Corporelle et découvrez votre statut pondéral.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vos informations")
            weight = st.number_input("Poids (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
            height = st.number_input("Taille (m)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
            
            if st.button("Calculer l'IMC", type="primary"):
                bmi, interpretation, color = calculate_bmi(weight, height)
                
                if bmi is not None:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                                color: white; padding: 2rem; border-radius: 1rem; 
                                box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 1rem 0;">
                        <h3 style="color: #3498db; font-size: 1.8rem; margin-bottom: 1rem; text-align: center;">
                            📊 Votre IMC : {bmi} kg/m²
                        </h3>
                        <h4 style="color: white; font-size: 1.3rem; text-align: center; font-weight: bold;">
                            {interpretation}
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Interprétation de l'IMC")
            st.markdown("""
            | IMC | Classification |
            |-----|---------------|
            | < 18.5 | Insuffisance pondérale |
            | 18.5 - 24.9 | Poids normal |
            | 25.0 - 29.9 | Surpoids |
            | ≥ 30.0 | Obésité |
            
            **Note :** L'IMC est un indicateur général et ne prend pas en compte la masse musculaire.
            """)
    
    # Calculateur de calories
    elif page == "🔥 Calculateur Calories":
        st.header("🔥 Calculateur de Calories")
        st.markdown("Estimez vos besoins caloriques quotidiens avec la formule Harris-Benedict.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vos informations")
            weight_cal = st.number_input("Poids (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, key="weight_cal")
            height_cal = st.number_input("Taille (cm)", min_value=100.0, max_value=250.0, value=175.0, step=1.0, key="height_cal")
            age = st.number_input("Âge (années)", min_value=10, max_value=100, value=30, step=1)
            gender = st.selectbox("Sexe", ["Homme", "Femme"])
            activity_level = st.selectbox(
                "Niveau d'activité",
                ["Sédentaire", "Légèrement actif", "Modérément actif", "Très actif", "Extrêmement actif"]
            )
            
            if st.button("Calculer les Calories", type="primary"):
                bmr, tdee = calculate_calories(weight_cal, height_cal/100, age, gender, activity_level)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                            color: white; padding: 2rem; border-radius: 1rem; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 1rem 0;">
                    <h3 style="color: #3498db; font-size: 1.8rem; margin-bottom: 1rem; text-align: center;">
                        🔥 Vos Besoins Caloriques
                    </h3>
                    <h4 style="color: white; font-size: 1.3rem; text-align: center; margin-bottom: 0.5rem;">
                        Métabolisme de Base (BMR) : {bmr} kcal/jour
                    </h4>
                    <h4 style="color: white; font-size: 1.3rem; text-align: center; font-weight: bold;">
                        Besoin Total (TDEE) : {tdee} kcal/jour
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                col_bmr, col_tdee = st.columns(2)
                
                with col_bmr:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 1.5rem; border-radius: 0.8rem; 
                                box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin: 0.5rem 0;">
                        <h4 style="color: white; font-size: 1.2rem; margin-bottom: 0.8rem;">🏠 Métabolisme de Base</h4>
                        <p style="color: white; font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>{bmr} kcal/jour</strong></p>
                        <p style="color: #f8f9fa; font-size: 0.9rem;">Calories nécessaires au repos</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_tdee:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                color: white; padding: 1.5rem; border-radius: 0.8rem; 
                                box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin: 0.5rem 0;">
                        <h4 style="color: white; font-size: 1.2rem; margin-bottom: 0.8rem;">⚡ Besoin Total</h4>
                        <p style="color: white; font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>{tdee} kcal/jour</strong></p>
                        <p style="color: #f8f9fa; font-size: 0.9rem;">Avec votre niveau d'activité</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Niveaux d'activité")
            st.markdown("""
            - **Sédentaire** : Peu ou pas d'exercice
            - **Légèrement actif** : Exercice léger 1-3 jours/semaine
            - **Modérément actif** : Exercice modéré 3-5 jours/semaine
            - **Très actif** : Exercice intense 6-7 jours/semaine
            - **Extrêmement actif** : Exercice très intense, travail physique
            
            **BMR** = Métabolisme de Base (calories au repos)
            **TDEE** = Total Daily Energy Expenditure (calories avec activité)
            """)
    
    # Générateur d'exercices
    elif page == "💪 Générateur d'Exercices":
        st.header("💪 Générateur d'Exercices")
        st.markdown("Générez un plan d'exercices personnalisé selon vos préférences.")
        
        # Charger les exercices
        exercises_df = load_exercises()
        
        if exercises_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Préférences d'entraînement")
                duration = st.selectbox("Durée de l'entraînement", [15, 30, 45, 60])
                difficulty = st.selectbox("Niveau de difficulté", ["Débutant", "Intermédiaire", "Avancé"])
                
                muscle_groups = st.multiselect(
                    "Groupes musculaires ciblés",
                    ["Poitrine", "Dos", "Épaules", "Bras", "Jambes", "Abdominaux", "Cardio"],
                    default=["Poitrine", "Dos", "Jambes"]
                )
                
                if st.button("Générer le Plan", type="primary"):
                    if muscle_groups:
                        exercises = generate_exercise_plan(exercises_df, duration, difficulty, muscle_groups)
                        
                        if exercises:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                        color: white; padding: 1.5rem; border-radius: 1rem; 
                                        box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 1rem 0; 
                                        text-align: center;">
                                <h3 style="color: white; font-size: 1.5rem; margin-bottom: 0.5rem;">
                                    🎉 Plan d'entraînement généré avec succès !
                                </h3>
                                <h4 style="color: #f8f9fa; font-size: 1.2rem;">
                                    Durée : {duration} minutes
                                </h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, exercise in enumerate(exercises, 1):
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            color: white; padding: 1.5rem; border-radius: 0.8rem; 
                                            box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin: 1rem 0;">
                                    <h4 style="color: white; font-size: 1.3rem; margin-bottom: 0.8rem; font-weight: bold;">
                                        {i}. {exercise['name']}
                                    </h4>
                                    <p style="color: #f8f9fa; margin-bottom: 0.5rem; font-size: 1rem;">
                                        <strong>Groupe musculaire :</strong> {exercise['muscle_group']}
                                    </p>
                                    <p style="color: #f8f9fa; margin-bottom: 0.5rem; font-size: 1rem;">
                                        <strong>Instructions :</strong> {exercise['instructions']}
                                    </p>
                                    <p style="color: #f8f9fa; margin-bottom: 0.5rem; font-size: 1rem;">
                                        <strong>Durée/Répétitions :</strong> {exercise['duration_reps']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        color: white; padding: 1.5rem; border-radius: 0.8rem; 
                                        box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin: 1rem 0;">
                                <h4 style="color: white; font-size: 1.2rem; margin-bottom: 0.8rem;">⚠️ Aucun exercice trouvé</h4>
                                <p style="color: #f8f9fa; font-size: 1rem;">Essayez de modifier vos critères de recherche</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    color: white; padding: 1.5rem; border-radius: 0.8rem; 
                                    box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin: 1rem 0;">
                            <h4 style="color: white; font-size: 1.2rem; margin-bottom: 0.8rem;">⚠️ Sélection requise</h4>
                            <p style="color: #f8f9fa; font-size: 1rem;">Veuillez sélectionner au moins un groupe musculaire</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Base de données des exercices")
                if st.checkbox("Afficher tous les exercices"):
                    st.dataframe(exercises_df)
        
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                        color: white; padding: 1.5rem; border-radius: 0.8rem; 
                        box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin: 1rem 0;">
                <h4 style="color: white; font-size: 1.2rem; margin-bottom: 0.8rem;">❌ Erreur de chargement</h4>
                <p style="color: #f8f9fa; font-size: 1rem;">Impossible de charger la base de données des exercices</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chatbot santé
    elif page == "🤖 Chatbot Santé":
        st.header("🤖 Chatbot Santé")
        st.markdown("Posez vos questions de santé et obtenez des conseils basiques.")
        
        # Initialiser l'historique du chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Zone de chat
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>Vous :</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>HealthMate :</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Zone de saisie
        user_input = st.text_input("Posez votre question de santé :", placeholder="Ex: Comment calculer mon IMC ?")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Envoyer", type="primary"):
                if user_input:
                    # Ajouter le message de l'utilisateur
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Obtenir la réponse du bot
                    bot_response = get_health_response(user_input)
                    
                    # Ajouter la réponse du bot
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": bot_response
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("Effacer l'historique"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Suggestions de questions
        st.subheader("💡 Questions suggérées")
        suggestions = [
            "Comment calculer mon IMC ?",
            "Combien de calories dois-je consommer ?",
            "Quels exercices pour les débutants ?",
            "Combien d'eau boire par jour ?",
            "Comment mieux dormir ?",
            "Comment gérer le stress ?"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": suggestion
                })
                
                bot_response = get_health_response(suggestion)
                
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": bot_response
                })
                
                st.rerun()
    
    # Suivi des Progrès
    elif page == "📈 Suivi des Progrès":
        st.header("📈 Suivi des Progrès")
        st.markdown("Enregistrez vos données quotidiennes et suivez votre évolution.")
        
        # Onglets pour différentes vues
        tab1, tab2, tab3 = st.tabs(["📝 Journal Quotidien", "📊 Graphiques", "📈 Statistiques"])
        
        with tab1:
            st.subheader("📝 Enregistrement Quotidien")
            
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input("Poids (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
                sleep_hours = st.number_input("Heures de sommeil", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
                energy_level = st.slider("Niveau d'énergie (1-10)", 1, 10, 7)
            
            with col2:
                mood = st.selectbox("Humeur", ["😊 Joyeux", "😐 Normal", "😔 Fatigué", "😠 Stressé"])
                notes = st.text_area("Notes personnelles", placeholder="Comment vous sentez-vous aujourd'hui ?")
            
            if st.button("Enregistrer les données", key="save_progress"):
                add_progress_entry(weight, sleep_hours, energy_level, mood, notes)
        
        with tab2:
            st.subheader("📊 Graphiques d'Évolution")
            
            if st.session_state.progress_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique de poids
                    weight_chart = create_progress_chart(st.session_state.progress_data, 'weight', 'Évolution du Poids')
                    if weight_chart:
                        st.plotly_chart(weight_chart, use_container_width=True)
                
                with col2:
                    # Graphique de sommeil
                    sleep_chart = create_progress_chart(st.session_state.progress_data, 'sleep_hours', 'Heures de Sommeil')
                    if sleep_chart:
                        st.plotly_chart(sleep_chart, use_container_width=True)
                
                # Graphique d'énergie
                energy_chart = create_progress_chart(st.session_state.progress_data, 'energy_level', 'Niveau d\'Énergie')
                if energy_chart:
                    st.plotly_chart(energy_chart, use_container_width=True)
            else:
                st.info("Aucune donnée de progression disponible. Commencez par enregistrer vos données quotidiennes.")
        
        with tab3:
            st.subheader("📈 Statistiques Mensuelles")
            
            if st.session_state.progress_data:
                df = pd.DataFrame(st.session_state.progress_data)
                df['date'] = pd.to_datetime(df['date'])
                
                # Statistiques générales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_weight = df['weight'].mean()
                    st.metric("Poids Moyen", f"{avg_weight:.1f} kg")
                
                with col2:
                    avg_sleep = df['sleep_hours'].mean()
                    st.metric("Sommeil Moyen", f"{avg_sleep:.1f} h")
                
                with col3:
                    avg_energy = df['energy_level'].mean()
                    st.metric("Énergie Moyenne", f"{avg_energy:.1f}/10")
                
                with col4:
                    total_entries = len(df)
                    st.metric("Jours Enregistrés", total_entries)
                
                # Tendances
                st.subheader("📊 Tendances")
                if len(df) > 1:
                    weight_trend = df['weight'].iloc[-1] - df['weight'].iloc[0]
                    sleep_trend = df['sleep_hours'].iloc[-1] - df['sleep_hours'].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Évolution Poids", f"{weight_trend:+.1f} kg", 
                                delta=f"{weight_trend:+.1f} kg depuis le début")
                    with col2:
                        st.metric("Évolution Sommeil", f"{sleep_trend:+.1f} h", 
                                delta=f"{sleep_trend:+.1f} h depuis le début")
            else:
                st.info("Aucune donnée disponible pour les statistiques.")
    
    # Nutrition Équilibrée
    elif page == "🥗 Nutrition Équilibrée":
        st.header("🥗 Nutrition Équilibrée")
        st.markdown("Calculez vos besoins nutritionnels et suivez votre alimentation.")
        
        tab1, tab2, tab3 = st.tabs(["🧮 Calculateur Macros", "📝 Journal Alimentaire", "📊 Analyse Nutritionnelle"])
        
        with tab1:
            st.subheader("🧮 Calculateur de Macronutriments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input("Poids (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, key="macro_weight")
                height = st.number_input("Taille (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, key="macro_height")
                age = st.number_input("Âge", min_value=10, max_value=100, value=30, key="macro_age")
            
            with col2:
                gender = st.selectbox("Sexe", ["Homme", "Femme"], key="macro_gender")
                activity_level = st.selectbox("Niveau d'activité", 
                    ["Sédentaire", "Légèrement actif", "Modérément actif", "Très actif", "Extrêmement actif"], 
                    key="macro_activity")
                goal = st.selectbox("Objectif", ["Maintien", "Perte de poids", "Prise de muscle"], key="macro_goal")
            
            if st.button("Calculer les Macros", key="calculate_macros"):
                macros = calculate_macros(weight, height, age, gender, activity_level, goal)
                
                st.success("Vos besoins nutritionnels ont été calculés !")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Calories", f"{macros['calories']} kcal")
                with col2:
                    st.metric("Protéines", f"{macros['proteins']} g")
                with col3:
                    st.metric("Glucides", f"{macros['carbs']} g")
                with col4:
                    st.metric("Lipides", f"{macros['fats']} g")
                
                # Graphique de répartition
                fig = go.Figure(data=[go.Pie(
                    labels=['Protéines', 'Glucides', 'Lipides'],
                    values=[macros['proteins']*4, macros['carbs']*4, macros['fats']*9],
                    hole=0.3
                )])
                fig.update_layout(title="Répartition des Macronutriments (Calories)")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📝 Journal Alimentaire")
            
            col1, col2 = st.columns(2)
            
            with col1:
                meal_type = st.selectbox("Type de repas", 
                    ["Petit-déjeuner", "Déjeuner", "Dîner", "Collation"], key="meal_type")
                food = st.text_input("Aliment/Plat", key="food_name")
            
            with col2:
                calories = st.number_input("Calories", min_value=0, max_value=2000, value=0, key="food_calories")
                proteins = st.number_input("Protéines (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, key="food_proteins")
            
            col3, col4 = st.columns(2)
            with col3:
                carbs = st.number_input("Glucides (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, key="food_carbs")
            with col4:
                fats = st.number_input("Lipides (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, key="food_fats")
            
            if st.button("Ajouter au Journal", key="add_meal"):
                if food and calories > 0:
                    add_nutrition_entry(meal_type, food, calories, proteins, carbs, fats)
                else:
                    st.error("Veuillez remplir au moins le nom de l'aliment et les calories.")
            
            # Affichage du journal du jour
            st.subheader("📋 Journal d'Aujourd'hui")
            today_entries = [entry for entry in st.session_state.nutrition_data 
                           if entry['date'] == datetime.now().strftime('%Y-%m-%d')]
            
            if today_entries:
                df_today = pd.DataFrame(today_entries)
                st.dataframe(df_today[['time', 'meal_type', 'food', 'calories', 'proteins', 'carbs', 'fats']], 
                           use_container_width=True)
            else:
                st.info("Aucun repas enregistré aujourd'hui.")
        
        with tab3:
            st.subheader("📊 Analyse Nutritionnelle")
            
            # Résumé quotidien
            daily_summary = get_daily_nutrition_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Calories", f"{daily_summary['calories']} kcal")
            with col2:
                st.metric("Protéines", f"{daily_summary['proteins']} g")
            with col3:
                st.metric("Glucides", f"{daily_summary['carbs']} g")
            with col4:
                st.metric("Lipides", f"{daily_summary['fats']} g")
            
            # Graphique d'évolution
            if st.session_state.nutrition_data:
                nutrition_chart = create_nutrition_chart(st.session_state.nutrition_data, days=7)
                if nutrition_chart:
                    st.plotly_chart(nutrition_chart, use_container_width=True)
            else:
                st.info("Aucune donnée nutritionnelle disponible.")
    
    # Santé Globale
    elif page == "💧 Santé Globale":
        st.header("💧 Santé Globale")
        st.markdown("Suivez votre hydratation, sommeil et bien-être général.")
        
        tab1, tab2, tab3 = st.tabs(["💧 Hydratation", "😴 Sommeil", "🧘 Bien-être"])
        
        with tab1:
            st.subheader("💧 Suivi Hydratation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Eau Consommée", f"{st.session_state.water_intake}ml", 
                         f"Objectif: {st.session_state.goals['water_goal']}ml")
                
                # Barre de progression
                progress = min(st.session_state.water_intake / st.session_state.goals['water_goal'], 1.0)
                st.progress(progress)
                
                if progress >= 1.0:
                    st.success("🎉 Objectif d'hydratation atteint !")
                elif progress >= 0.8:
                    st.warning("💧 Presque à l'objectif !")
                else:
                    st.info("💧 Continuez à boire de l'eau !")
            
            with col2:
                water_to_add = st.number_input("Eau à ajouter (ml)", min_value=0, max_value=1000, value=250, step=50)
                
                if st.button("Ajouter de l'eau", key="add_water"):
                    st.session_state.water_intake += water_to_add
                    st.success(f"{water_to_add}ml d'eau ajoutée !")
                    st.rerun()
                
                if st.button("Réinitialiser", key="reset_water"):
                    st.session_state.water_intake = 0
                    st.success("Hydratation réinitialisée !")
                    st.rerun()
        
        with tab2:
            st.subheader("😴 Suivi du Sommeil")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sleep_hours = st.number_input("Heures de sommeil", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
                sleep_quality = st.selectbox("Qualité du sommeil", 
                    ["😴 Excellent", "😊 Bon", "😐 Moyen", "😔 Mauvais"])
                sleep_notes = st.text_area("Notes sur le sommeil", placeholder="Comment avez-vous dormi ?")
            
            with col2:
                if st.button("Enregistrer le sommeil", key="save_sleep"):
                    sleep_entry = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'hours': sleep_hours,
                        'quality': sleep_quality,
                        'notes': sleep_notes
                    }
                    st.session_state.sleep_data.append(sleep_entry)
                    st.success("Données de sommeil enregistrées !")
                
                # Statistiques de sommeil
                if st.session_state.sleep_data:
                    df_sleep = pd.DataFrame(st.session_state.sleep_data)
                    avg_sleep = df_sleep['hours'].mean()
                    st.metric("Sommeil Moyen", f"{avg_sleep:.1f} h")
        
        with tab3:
            st.subheader("🧘 Bien-être et Gestion du Stress")
            
            st.markdown("### Techniques de Relaxation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🫁 Respiration 4-7-8")
                st.markdown("""
                1. Inspirez par le nez pendant 4 secondes
                2. Retenez votre souffle pendant 7 secondes
                3. Expirez par la bouche pendant 8 secondes
                4. Répétez 4 cycles
                """)
                
                if st.button("Commencer la Respiration", key="breathing"):
                    st.success("💨 Exercice de respiration démarré ! Respirez calmement...")
            
            with col2:
                st.markdown("#### 🧘 Méditation Guidée")
                st.markdown("""
                - Asseyez-vous confortablement
                - Fermez les yeux
                - Concentrez-vous sur votre respiration
                - Laissez passer les pensées sans les juger
                - Pratiquez pendant 5-10 minutes
                """)
                
                if st.button("Commencer la Méditation", key="meditation"):
                    st.success("🧘 Session de méditation démarrée !")
            
            # Quiz bien-être
            st.markdown("### 📋 Quiz Bien-être")
            
            stress_level = st.slider("Niveau de stress (1-10)", 1, 10, 5)
            energy_today = st.slider("Énergie aujourd'hui (1-10)", 1, 10, 7)
            mood_today = st.selectbox("Humeur générale", ["😊 Très bien", "😐 Bien", "😔 Moyen", "😠 Difficile"])
            
            if st.button("Évaluer mon Bien-être", key="wellness_quiz"):
                score = (10 - stress_level) + energy_today
                if mood_today == "😊 Très bien":
                    score += 3
                elif mood_today == "😐 Bien":
                    score += 2
                elif mood_today == "😔 Moyen":
                    score += 1
                
                st.markdown(f"### Score de Bien-être: {score}/23")
                
                if score >= 18:
                    st.success("🌟 Excellent ! Vous êtes en pleine forme !")
                elif score >= 14:
                    st.info("👍 Bien ! Continuez sur cette lancée !")
                elif score >= 10:
                    st.warning("⚠️ Moyen. Prenez soin de vous !")
                else:
                    st.error("🚨 Faible. Consultez un professionnel de santé si nécessaire.")
    
    # Mes Objectifs
    elif page == "🎯 Mes Objectifs":
        st.header("🎯 Mes Objectifs Personnels")
        st.markdown("Définissez et suivez vos objectifs de santé personnalisés.")
        
        # Section de définition des objectifs
        st.subheader("📝 Définir Mes Objectifs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Objectifs de Poids")
            weight_goal = st.number_input("Poids cible (kg)", min_value=30.0, max_value=300.0, 
                                        value=float(st.session_state.goals['weight_goal']) if st.session_state.goals['weight_goal'] > 0 else 70.0, 
                                        step=0.1, key="weight_goal_input")
            
            st.markdown("#### 💧 Objectif d'Hydratation")
            water_goal = st.number_input("Objectif d'eau quotidien (ml)", min_value=500, max_value=5000, 
                                       value=st.session_state.goals['water_goal'], step=100, key="water_goal_input")
        
        with col2:
            st.markdown("#### 😴 Objectif de Sommeil")
            sleep_goal = st.number_input("Heures de sommeil par nuit", min_value=4.0, max_value=12.0, 
                                       value=float(st.session_state.goals['sleep_goal']), step=0.5, key="sleep_goal_input")
            
            st.markdown("#### 🏃 Objectif d'Exercice")
            exercise_goal = st.number_input("Minutes d'exercice par semaine", min_value=30, max_value=1000, 
                                          value=st.session_state.goals['exercise_goal'], step=15, key="exercise_goal_input")
        
        # Bouton pour sauvegarder les objectifs
        if st.button("💾 Sauvegarder Mes Objectifs", key="save_goals"):
            st.session_state.goals = {
                'weight_goal': weight_goal,
                'water_goal': water_goal,
                'sleep_goal': sleep_goal,
                'exercise_goal': exercise_goal
            }
            st.success("🎉 Vos objectifs ont été sauvegardés !")
            st.rerun()
        
        st.markdown("---")
        
        # Section de suivi des objectifs
        st.subheader("📊 Suivi de Mes Objectifs")
        
        goals = st.session_state.goals
        
        # Métriques des objectifs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if goals['weight_goal'] > 0:
                if st.session_state.progress_data:
                    current_weight = st.session_state.progress_data[-1]['weight']
                    weight_diff = current_weight - goals['weight_goal']
                    st.metric("🎯 Poids Cible", f"{goals['weight_goal']} kg", 
                             delta=f"{weight_diff:+.1f} kg" if weight_diff != 0 else "Objectif atteint !")
                else:
                    st.metric("🎯 Poids Cible", f"{goals['weight_goal']} kg", "Non défini")
            else:
                st.metric("🎯 Poids Cible", "Non défini")
        
        with col2:
            water_progress = min(st.session_state.water_intake / goals['water_goal'], 1.0)
            st.metric("💧 Hydratation", f"{int(water_progress*100)}%", 
                     f"Objectif: {goals['water_goal']}ml")
        
        with col3:
            if st.session_state.progress_data:
                avg_sleep = sum(entry['sleep_hours'] for entry in st.session_state.progress_data[-7:]) / min(len(st.session_state.progress_data), 7)
                sleep_diff = avg_sleep - goals['sleep_goal']
                st.metric("😴 Sommeil", f"{avg_sleep:.1f}h", 
                         delta=f"{sleep_diff:+.1f}h" if sleep_diff != 0 else "Objectif atteint !")
            else:
                st.metric("😴 Sommeil", "Non enregistré", f"Objectif: {goals['sleep_goal']}h")
        
        with col4:
            # Pour l'exercice, on simule des données (dans une vraie app, ceci viendrait d'un tracker)
            exercise_progress = 0  # À remplacer par des vraies données d'exercice
            st.metric("🏃 Exercice", f"{exercise_progress}min", 
                     f"Objectif: {goals['exercise_goal']}min/semaine")
        
        # Graphiques de progression vers les objectifs
        st.subheader("📈 Progression vers Mes Objectifs")
        
        if st.session_state.progress_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique de progression du poids
                if goals['weight_goal'] > 0:
                    df = pd.DataFrame(st.session_state.progress_data)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['date'], 
                        y=df['weight'],
                        mode='lines+markers',
                        name='Poids Actuel',
                        line=dict(color='#FF6B6B')
                    ))
                    fig.add_hline(y=goals['weight_goal'], 
                                 line_dash="dash", 
                                 line_color="green",
                                 annotation_text=f"Objectif: {goals['weight_goal']} kg")
                    
                    fig.update_layout(
                        title="Progression vers l'Objectif de Poids",
                        xaxis_title="Date",
                        yaxis_title="Poids (kg)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Graphique de progression de l'hydratation
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = st.session_state.water_intake,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Hydratation (ml)"},
                    delta = {'reference': goals['water_goal']},
                    gauge = {
                        'axis': {'range': [None, goals['water_goal']]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, goals['water_goal']*0.5], 'color': "lightgray"},
                            {'range': [goals['water_goal']*0.5, goals['water_goal']*0.8], 'color': "yellow"},
                            {'range': [goals['water_goal']*0.8, goals['water_goal']], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': goals['water_goal']
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Conseils personnalisés
        st.subheader("💡 Conseils Personnalisés")
        
        advice = []
        
        # Conseils basés sur les objectifs
        if goals['weight_goal'] > 0 and st.session_state.progress_data:
            current_weight = st.session_state.progress_data[-1]['weight']
            if current_weight > goals['weight_goal']:
                advice.append("🎯 **Poids** : Vous êtes au-dessus de votre objectif. Consultez la section Nutrition pour des conseils de perte de poids.")
            elif current_weight < goals['weight_goal']:
                advice.append("🎯 **Poids** : Vous êtes en dessous de votre objectif. Assurez-vous de manger suffisamment.")
        
        water_progress = st.session_state.water_intake / goals['water_goal']
        if water_progress < 0.8:
            advice.append("💧 **Hydratation** : Buvez plus d'eau ! Votre objectif est de " + str(goals['water_goal']) + "ml par jour.")
        elif water_progress >= 1.0:
            advice.append("💧 **Hydratation** : Excellent ! Vous avez atteint votre objectif d'hydratation.")
        
        if st.session_state.progress_data:
            recent_sleep = st.session_state.progress_data[-1]['sleep_hours']
            if recent_sleep < goals['sleep_goal']:
                advice.append("😴 **Sommeil** : Essayez de dormir au moins " + str(goals['sleep_goal']) + " heures par nuit.")
        
        if advice:
            for tip in advice:
                st.info(tip)
        else:
            st.success("🎉 Excellent ! Vous êtes sur la bonne voie pour atteindre tous vos objectifs !")
        
        # Section de motivation
        st.subheader("🌟 Motivation")
        
        # Calcul du score de motivation
        motivation_score = 0
        total_goals = 4
        
        if goals['weight_goal'] > 0 and st.session_state.progress_data:
            current_weight = st.session_state.progress_data[-1]['weight']
            if abs(current_weight - goals['weight_goal']) <= 2:
                motivation_score += 1
        
        if water_progress >= 0.8:
            motivation_score += 1
        
        if st.session_state.progress_data:
            recent_sleep = st.session_state.progress_data[-1]['sleep_hours']
            if abs(recent_sleep - goals['sleep_goal']) <= 1:
                motivation_score += 1
        
        # Pour l'exercice, on considère qu'il est atteint si l'utilisateur utilise l'app
        motivation_score += 1
        
        motivation_percentage = (motivation_score / total_goals) * 100
        
        st.markdown(f"### Score de Motivation: {motivation_percentage:.0f}%")
        
        if motivation_percentage >= 75:
            st.success("🌟 **Excellent !** Vous êtes très motivé et sur la bonne voie !")
        elif motivation_percentage >= 50:
            st.info("👍 **Bien !** Continuez vos efforts, vous progressez !")
        else:
            st.warning("💪 **Courage !** Il y a encore du travail à faire, mais vous pouvez y arriver !")
    
    # Dashboard
    elif page == "📊 Dashboard":
        st.header("📊 Dashboard Principal")
        st.markdown("Vue d'ensemble de votre santé et de vos progrès.")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.progress_data:
                latest_weight = st.session_state.progress_data[-1]['weight']
                st.metric("Poids Actuel", f"{latest_weight} kg")
            else:
                st.metric("Poids Actuel", "Non enregistré")
        
        with col2:
            daily_summary = get_daily_nutrition_summary()
            st.metric("Calories Aujourd'hui", f"{daily_summary['calories']} kcal")
        
        with col3:
            progress = min(st.session_state.water_intake / st.session_state.goals['water_goal'], 1.0)
            st.metric("Hydratation", f"{int(progress*100)}%")
        
        with col4:
            if st.session_state.progress_data:
                latest_energy = st.session_state.progress_data[-1]['energy_level']
                st.metric("Énergie", f"{latest_energy}/10")
            else:
                st.metric("Énergie", "Non enregistrée")
        
        # Graphiques de synthèse
        if st.session_state.progress_data:
            col1, col2 = st.columns(2)
            
            with col1:
                weight_chart = create_progress_chart(st.session_state.progress_data, 'weight', 'Évolution du Poids')
                if weight_chart:
                    st.plotly_chart(weight_chart, use_container_width=True)
            
            with col2:
                energy_chart = create_progress_chart(st.session_state.progress_data, 'energy_level', 'Niveau d\'Énergie')
                if energy_chart:
                    st.plotly_chart(energy_chart, use_container_width=True)
        
        # Recommandations
        st.subheader("💡 Recommandations Personnalisées")
        
        recommendations = []
        
        # Recommandations basées sur les données
        if st.session_state.water_intake < st.session_state.goals['water_goal'] * 0.8:
            recommendations.append("💧 Buvez plus d'eau pour atteindre votre objectif d'hydratation")
        
        if st.session_state.progress_data:
            latest_sleep = st.session_state.progress_data[-1]['sleep_hours']
            if latest_sleep < 7:
                recommendations.append("😴 Essayez de dormir au moins 7 heures par nuit")
        
        daily_summary = get_daily_nutrition_summary()
        if daily_summary['calories'] < 1200:
            recommendations.append("🍎 Assurez-vous de manger suffisamment de calories")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("🎉 Excellent ! Vous suivez bien vos objectifs de santé !")

if __name__ == "__main__":
    main()
