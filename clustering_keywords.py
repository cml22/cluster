import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

# Charger le fichier CSV
uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())  # Afficher les premières lignes du fichier

        # Vérification si la colonne existe
        if 'Requêtes les plus fréquentes' not in df.columns:
            st.error("La colonne 'Requêtes les plus fréquentes' n'existe pas dans le fichier CSV.")
        else:
            # Sélectionner le nombre de clusters avec un curseur
            num_clusters = st.slider("Sélectionnez le nombre de clusters", min_value=2, max_value=20, value=5, step=1)

            # Vectorisation TF-IDF sans stop words
            try:
                vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")  # Pas de stop words
                X = vectorizer.fit_transform(df['Requêtes les plus fréquentes'])
            except Exception as e:
                st.error(f"Erreur lors de la vectorisation TF-IDF : {str(e)}")
                st.stop()

            # Clustering avec KMeans
            try:
                model = KMeans(n_clusters=num_clusters, random_state=42)
                model.fit(X)
                clusters = model.labels_
                df['Cluster'] = clusters
            except Exception as e:
                st.error(f"Erreur lors du clustering : {str(e)}")
                st.stop()

            # Calculer le nombre de mots par cluster
            cluster_word_count = Counter(clusters)

            # Afficher les clusters
            st.subheader("Clusters et mots associés")
            for cluster in range(num_clusters):
                st.write(f"**Cluster {cluster}:**")
                words = df[df['Cluster'] == cluster]['Requêtes les plus fréquentes'].tolist()
                st.write(", ".join(words))

            # Visualisation des clusters
            fig, ax = plt.subplots()
            ax.bar(cluster_word_count.keys(), cluster_word_count.values())
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Nombre de mots")
            ax.set_title("Nombre de mots par cluster")
            st.pyplot(fig)

            # Filtrage des clusters
            st.subheader("Filtrer les clusters")
            clusters_to_keep = st.multiselect("Sélectionnez les clusters à garder", list(range(num_clusters)), default=list(range(num_clusters)))

            # Filtrer le DataFrame selon les clusters sélectionnés
            filtered_df = df[df['Cluster'].isin(clusters_to_keep)]
            st.write("DataFrame filtré:")
            st.write(filtered_df)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {str(e)}")
