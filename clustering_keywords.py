import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re

# Charger le fichier CSV
uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())  # Afficher les premières lignes du fichier

        # Vérification si la colonne existe bien
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
                df['Cluster'] = model.labels_
            except Exception as e:
                st.error(f"Erreur lors du clustering KMeans : {str(e)}")
                st.stop()

            # Déterminer le sujet parent par cluster
            def get_parent_term(cluster_num):
                cluster_terms = df[df['Cluster'] == cluster_num]['Requêtes les plus fréquentes']
                # Récupérer les mots et les normaliser
                all_words = ' '.join(cluster_terms).lower()
                # Supprimer les caractères spéciaux et diviser en mots
                words = re.findall(r'\b\w+\b', all_words)
                # Compter la fréquence des mots
                word_counts = Counter(words)
                # Filtrer les mots peu significatifs (par exemple, moins de 3 caractères)
                filtered_words = {word: count for word, count in word_counts.items() if len(word) > 2}
                # Retourner le mot avec la plus haute fréquence ou un groupe de mots similaires
                if filtered_words:
                    # Utiliser Counter pour trouver le mot le plus commun
                    most_common_word, _ = Counter(filtered_words).most_common(1)[0]
                    return most_common_word
                return "Inconnu"

            df['Nom du Cluster'] = df['Cluster'].apply(get_parent_term)

            # Afficher les clusters et leur nom parent
            st.write("Clusters créés :")
            st.write(df[['Requêtes les plus fréquentes', 'Cluster', 'Nom du Cluster']])

            # Exporter le résultat en CSV
            csv_export = df.to_csv(index=False)
            st.download_button("Télécharger le fichier clusterisé", csv_export, "clusters.csv", "text/csv")
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {str(e)}")
else:
    st.write("Veuillez téléverser un fichier CSV.")
