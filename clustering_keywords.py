import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Upload or bulk input
st.title("Clustering de mots-clés")

# Option 1: Upload a CSV file
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Vérifie si l'une des colonnes attendues est présente
    expected_columns = ['Requêtes les plus fréquentes', 'Clics', 'Impressions', 'CTR', 'Position']
    if not any(col in df.columns for col in expected_columns):
        st.error("Le fichier CSV doit contenir l'une des colonnes suivantes : " + ", ".join(expected_columns))
    else:
        # Utiliser la colonne des requêtes
        if 'Requêtes les plus fréquentes' in df.columns:
            keywords = df['Requêtes les plus fréquentes'].dropna()  # Remove any empty values
        else:
            st.error("La colonne 'Requêtes les plus fréquentes' est manquante dans le fichier CSV.")
else:
    # Option 2: Bulk input
    keywords_input = st.text_area("Entrer les mots-clés (séparés par des virgules)")
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]  # Clean and filter keywords
    else:
        keywords = []  # Initialize as empty if no input

# Use
