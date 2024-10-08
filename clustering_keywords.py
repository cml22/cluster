import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Fonction de prétraitement
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words and len(word) > 2]
    return ' '.join(words)

# Charger le fichier CSV
uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        if 'Requêtes les plus fréquentes' not in df.columns:
            st.error("La colonne 'Requêtes les plus fréquentes' n'existe pas dans le fichier CSV.")
        else:
            df['Processed'] = df['Requêtes les plus fréquentes'].apply(preprocess_text)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['Processed'])

            # Sélectionner le nombre de clusters
            num_clusters = st.slider("Nombre de clusters souhaité :", min_value=2, max_value=20, value=5)

            # KMeans pour le clustering
            model = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = model.fit_predict(X)
            df['Cluster'] = clusters

            st.subheader("Sujets Parents des Clusters")

            cluster_info = {}

            for cluster in set(clusters):
                words = df[df['Cluster'] == cluster]['Requêtes les plus fréquentes'].tolist()
                
                # Compter la fréquence des mots pour trouver le sujet parent
                word_freq = Counter()
                for word in words:
                    for w in word.split():
                        word_freq[w] += 1
                
                # Trouver les mots les plus communs
                if word_freq:
                    parent_topic = ' '.join([word for word, count in word_freq.most_common(3)])  # Les 3 mots les plus fréquents
                    cluster_info[cluster] = {
                        'topic': parent_topic,
                        'keywords': words,
                        'size': len(words)
                    }
                    st.write(f"**Cluster {cluster} :** {parent_topic} (Mots associés: {len(words)})")
                    if st.button(f"Afficher les mots-clés associés au Cluster {cluster}"):
                        st.write(", ".join(words))
                else:
                    st.write("- Aucune donnée disponible.")

                st.markdown("---")  # Séparation entre les clusters

            # Visualisation des clusters
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X.toarray())
            df['PCA_1'] = X_pca[:, 0]
            df['PCA_2'] = X_pca[:, 1]

            plt.figure(figsize=(10, 6))
            for cluster, info in cluster_info.items():
                plt.scatter(info['size'], cluster, s=info['size']*10, alpha=0.5, label=info['topic'])  # Taille proportionnelle aux mots-clés

            plt.title("Visualisation des Clusters")
            plt.xlabel("Taille des mots-clés")
            plt.ylabel("Clusters")
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {str(e)}")
