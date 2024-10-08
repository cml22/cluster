import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Fonction pour récupérer le mot-clé principal de chaque cluster
def get_cluster_name(df, labels):
    cluster_names = {}
    for cluster_id in set(labels):
        if cluster_id != -1:  # Exclude outliers
            cluster_data = df[labels == cluster_id]
            # Calcul du mot-clé principal en fonction des occurrences de TF-IDF
            main_keyword = cluster_data["Requêtes les plus fréquentes"].iloc[0]
            cluster_names[cluster_id] = main_keyword
        else:
            cluster_names[cluster_id] = "Outliers"
    return cluster_names

# Fonction pour uploader le fichier
uploaded_file = st.file_uploader("Téléversez le fichier CSV contenant les requêtes", type=["csv"])

if uploaded_file is not None:
    # Lecture du fichier CSV
    df = pd.read_csv(uploaded_file)

    # Vérification que les colonnes attendues existent
    if "Requêtes les plus fréquentes" in df.columns and "Clics" in df.columns and "Impressions" in df.columns:
        
        # Vectorisation des requêtes
        vectorizer = TfidfVectorizer(stop_words='french')
        X = vectorizer.fit_transform(df['Requêtes les plus fréquentes'])

        # Application du clustering DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = dbscan.fit_predict(X)

        # Ajouter les labels de clusters dans le DataFrame
        df['Cluster'] = labels

        # Récupérer le nom des clusters basés sur le mot-clé le plus représenté
        cluster_names = get_cluster_name(df, labels)

        # Afficher les clusters avec leur nom
        df['Nom du Cluster'] = df['Cluster'].map(cluster_names)

        # Affichage des résultats
        st.write("Résultats du clustering :")
        st.dataframe(df)

        # Visualisation des clusters
        fig, ax = plt.subplots()
        unique_labels = set(labels)

        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]  # Black for outliers

            class_member_mask = (labels == k)

            xy = X.toarray()[class_member_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        plt.title('Représentation des clusters de requêtes')
        st.pyplot(fig)

    else:
        st.error("Les colonnes 'Requêtes les plus fréquentes', 'Clics', 'Impressions' sont manquantes.")
else:
    st.info("Veuillez téléverser un fichier CSV.")
