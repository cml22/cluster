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
