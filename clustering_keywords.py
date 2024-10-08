import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk

# Assure-toi que NLTK est prêt (ajoute ceci si nécessaire)
nltk.download('punkt')

# Titre de l'application
st.title("Clustering de Mots-Clés")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
    st.write("Aperçu des données :", df.head())

    # Affiche les colonnes disponibles
    st.write("Colonnes disponibles :", df.columns.tolist())  # Affiche sous forme de liste

    # Vérifie si la colonne 'Requêtes les plus fréquentes' existe
    if 'Requêtes les plus fréquentes' in df.columns:
        # Traitement et clustering des données
        st.write("Traitement en cours...")

        # Exemple d'utilisation de TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['Requêtes les plus fréquentes'])

        # Exemple de clustering avec DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(X.toarray())

        # Ajouter une nouvelle colonne pour le cluster dans le DataFrame
        df['Cluster'] = clustering.labels_

        # Afficher les résultats
        st.write("Résultats du clustering :", df)
    else:
        st.error("La colonne 'Requêtes les plus fréquentes' n'existe pas dans le fichier CSV. Veuillez vérifier le nom de la colonne.")
else:
    st.warning("Veuillez télécharger un fichier CSV contenant les requêtes.")
