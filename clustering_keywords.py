import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Charger le fichier CSV
uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Afficher les premières lignes du fichier

    # Sélectionner le nombre de clusters avec un curseur
    num_clusters = st.slider("Sélectionnez le nombre de clusters", min_value=2, max_value=20, value=5, step=1)

    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(stop_words='french', token_pattern=r"(?u)\b\w\w+\b")
    X = vectorizer.fit_transform(df['Requêtes les plus fréquentes'])

    # Clustering avec KMeans
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)

    # Ajouter les labels de clusters au dataframe
    df['Cluster'] = model.labels_

    # Déterminer le mot-clé dominant par cluster
    def get_dominant_term(cluster_num):
        cluster_terms = df[df['Cluster'] == cluster_num]['Requêtes les plus fréquentes']
        all_words = ' '.join(cluster_terms).split()
        most_common_word, _ = Counter(all_words).most_common(1)[0]  # Récupérer le mot le plus fréquent
        return most_common_word

    df['Nom du Cluster'] = df['Cluster'].apply(get_dominant_term)

    # Afficher les clusters et leur nom dominant
    st.write("Clusters créés :")
    st.write(df[['Requêtes les plus fréquentes', 'Cluster', 'Nom du Cluster']])

    # Exporter le résultat en CSV
    csv_export = df.to_csv(index=False)
    st.download_button("Télécharger le fichier clusterisé", csv_export, "clusters.csv", "text/csv")

else:
    st.write("Veuillez téléverser un fichier CSV.")
