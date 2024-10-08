import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns

nltk.download('stopwords')
nltk.download('wordnet')

# Fonction de prétraitement
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))
    # Tokenisation et lemmatisation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

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
            # Prétraitement
            df['Processed'] = df['Requêtes les plus fréquentes'].apply(preprocess_text)

            # Vectorisation TF-IDF
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['Processed'])

            # Clustering avec DBSCAN
            model = DBSCAN(eps=0.5, min_samples=2, metric='cosine')  # Ajustez les paramètres selon vos données
            clusters = model.fit_predict(X)
            df['Cluster'] = clusters

            # Vérifier les clusters formés
            unique_clusters = set(clusters)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)  # Supprimer le bruit

            st.subheader("Clusters et mots associés")
            for cluster in unique_clusters:
                st.write(f"**Cluster {cluster}:**")
                words = df[df['Cluster'] == cluster]['Requêtes les plus fréquentes'].tolist()
                st.write(", ".join(words))

            # Visualisation des clusters
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X.toarray())
            df['PCA_1'] = X_pca[:, 0]
            df['PCA_2'] = X_pca[:, 1]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PCA_1', y='PCA_2', hue='Cluster', data=df, palette='viridis', legend='full')
            plt.title("Visualisation des Clusters avec DBSCAN")
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {str(e)}")
