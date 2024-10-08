import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Upload or bulk input
st.title("Clustering de mots-clés")

# Option 1: Upload a CSV file
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'query' not in df.columns:
        st.error("La colonne 'query' est manquante dans le fichier CSV.")
    else:
        keywords = df['query'].dropna()  # Remove any empty values
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
    # Vectorize the keywords using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(keywords)

    # Clustering using KMeans
    num_clusters = st.slider("Nombre de clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Add cluster labels to the data
    df['cluster'] = kmeans.labels_

    # Export result as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Télécharger les clusters", data=csv, file_name="clusters_keywords.csv", mime='text/csv')

    # Display the clustered data
    st.write(df)
else:
    st.warning("Aucun mot-clé disponible pour le clustering.")
