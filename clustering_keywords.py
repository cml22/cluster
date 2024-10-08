import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import re

# Fonction pour normaliser et appliquer le stemming aux mots-clés
def stem_keywords(keywords):
    ps = PorterStemmer()
    # Normalisation
    keywords = [re.sub(r'\d+', '', k) for k in keywords]  # Supprime les chiffres
    keywords = [k.strip().lower() for k in keywords]  # Mise en minuscule
    # Application du stemming
    stemmed_keywords = [' '.join([ps.stem(word) for word in k.split()]) for k in keywords]
    return stemmed_keywords

# Fonction pour préparer les données des clusters
def prepare_cluster_data(df):
    cluster_summary = df.groupby('cluster').agg({
        'Clics': 'sum',
        'Impressions': 'sum',
        'Requêtes les plus fréquentes': 'count'
    }).reset_index()

    cluster_summary.columns = ['Cluster', 'Total Clics', 'Total Impressions', 'Nombre de Mots-Clés']
    return cluster_summary

# Upload or bulk input
st.title("Clustering de mots-clés")

# Option 1: Upload a CSV file
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Données chargées :")
    st.write(df.head())  # Afficher les premières lignes du DataFrame pour vérification

    # Vérifie si l'une des colonnes attendues est présente
    expected_columns = ['Requêtes les plus fréquentes', 'Clics', 'Impressions', 'CTR', 'Position']
    if not any(col in df.columns for col in expected_columns):
        st.error("Le fichier CSV doit contenir l'une des colonnes suivantes : " + ", ".join(expected_columns))
    else:
        # Utiliser la colonne des requêtes
        if 'Requêtes les plus fréquentes' in df.columns:
            keywords = df['Requêtes les plus fréquentes'].dropna()  # Remove any empty values
            st.write("Mots-clés détectés :")
            st.write(keywords.head())  # Afficher les premières valeurs des mots-clés
        else:
            st.error("La colonne 'Requêtes les plus fréquentes' est manquante dans le fichier CSV.")
else:
    # Option 2: Bulk input
    keywords_input = st.text_area("Entrer les mots-clés (séparés par des virgules)")
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]  # Clean and filter keywords
    else:
        keywords = []  # Initialize as empty if no input

# Use .empty to check if the keywords are empty
if isinstance(keywords, pd.Series) and not keywords.empty:
    # Continue with the clustering process
    # Normaliser et appliquer le stemming
    stemmed_keywords = stem_keywords(keywords)

    # Vectoriser les mots-clés en utilisant TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(stemmed_keywords)

    # Clustering using DBSCAN
    db = DBSCAN(eps=0.5, min_samples=2).fit(X.toarray())
    df['cluster'] = db.labels_

    # Filtrer les clusters non attribués (-1)
    if len(df['cluster'].unique()) > 1:  # Plus d'un cluster
        # Préparer les données des clusters
        cluster_data = prepare_cluster_data(df)

        # Visualisation sous forme de bulles
        plt.figure(figsize=(10, 6))
        plt.scatter(cluster_data['Nombre de Mots-Clés'], cluster_data['Total Clics'], 
                    s=cluster_data['Total Impressions'] / 100, alpha=0.5)

        # Ajouter les noms de clusters avec annotations visuelles
        for i in range(len(cluster_data)):
            plt.annotate(f"Cluster {cluster_data['Cluster'][i]} (N={cluster_data['Nombre de Mots-Clés'][i]})", 
                         (cluster_data['Nombre de Mots-Clés'][i], cluster_data['Total Clics'][i]),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')

        plt.title('Visualisation des Clusters')
        plt.xlabel('Nombre de Mots-Clés')
        plt.ylabel('Total Clics')
        plt.grid()

        st.pyplot(plt)

        # Export result as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Télécharger les clusters", data=csv, file_name="clusters_keywords.csv", mime='text/csv')

        # Display the clustered data
        st.write(df)
    else:
        st.warning("Aucun cluster valide trouvé. Essayez d'ajuster les paramètres de DBSCAN.")
else:
    st.warning("Aucun mot-clé disponible pour le clustering.")
