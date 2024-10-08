import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk

nltk.download('punkt')

def generate_cluster_name(cluster_keywords):
    if any("invitation" in keyword for keyword in cluster_keywords):
        return "Guide complet sur les invitations de mariage"
    elif any("chaussure" in keyword for keyword in cluster_keywords):
        return "Chaussures pour différents types de jeans"
    else:
        return "Autre"

# Titre de l'application
st.title("Clustering de Mots-Clés")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Applique le TF-IDF et le clustering
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Requêtes les plus fréquentes'])
    db = DBSCAN(eps=0.5, min_samples=2).fit(X.toarray())

    # Ajoute les clusters au DataFrame
    df['Cluster'] = db.labels_

    # Nommer les clusters
    df['Nom du Cluster'] = df.groupby('Cluster')['Requêtes les plus fréquentes'].transform(generate_cluster_name)

    # Regroupe les données par cluster et calcule les sommes
    summary = df.groupby('Nom du Cluster').agg({
        'Clics': 'sum',
        'Impressions': 'sum',
        'CTR': 'mean',
        'Position': 'mean'
    }).reset_index()

    st.write(summary)
