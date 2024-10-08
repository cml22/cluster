import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')

# Charger le fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Vérifier la colonne 'Requêtes les plus fréquentes'
    if 'Requêtes les plus fréquentes' not in df.columns:
        st.error("Le fichier CSV doit contenir la colonne 'Requêtes les plus fréquentes'.")
    else:
        # Vectorisation des requêtes
        vectorizer = CountVectorizer(stop_words=stopwords.words('french'))
        X = vectorizer.fit_transform(df['Requêtes les plus fréquentes'])

        # Clustering avec KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X)
        df['Nom du Cluster'] = kmeans.labels_

        # Trouver le mot-clé le plus représenté pour nommer les clusters
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        cluster_names = {}

        for i in range(5):
            top_keywords = [terms[ind] for ind in order_centroids[i, :5]]
            cluster_names[i] = ', '.join(top_keywords)

        df['Nom du Cluster'] = df['Nom du Cluster'].map(cluster_names)

        # Afficher le résultat
        st.write(df)

        # Visualisation
        plt.figure(figsize=(10, 6))
        df['Nom du Cluster'].value_counts().plot(kind='bar')
        plt.title('Répartition des clusters')
        plt.xlabel('Nom du Cluster')
        plt.ylabel('Nombre de requêtes')
        st.pyplot(plt)
