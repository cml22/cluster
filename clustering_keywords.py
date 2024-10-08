import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
try:
    df = pd.read_csv('Requêtes.csv')
except FileNotFoundError:
    st.error("Le fichier 'Requêtes.csv' est introuvable.")
    st.stop()

# Afficher les données initiales
st.write("Données initiales :")
st.dataframe(df)

# Vérifier et convertir les colonnes numériques
numeric_columns = ['Clics', 'Impressions', 'CTR', 'Position']
for col in numeric_columns:
    if df[col].dtype != 'float64' and df[col].dtype != 'int64':
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertit en numérique et remplace les erreurs par NaN

# Remplacer les NaN par 0
df.fillna(0, inplace=True)

# Appliquer le stemming et le TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='french', tokenizer=lambda x: x.split())
tfidf_matrix = tfidf.fit_transform(df['Requête'])

# Appliquer DBSCAN pour le clustering
dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
labels = dbscan.fit_predict(tfidf_matrix.toarray())

# Ajouter les labels de cluster au DataFrame
df['Nom du Cluster'] = ['Cluster ' + str(label) if label != -1 else 'Noise' for label in labels]

# Résumé des clusters
summary = df.groupby('Nom du Cluster').agg({
    'Clics': 'sum',
    'Impressions': 'sum',
    'CTR': 'mean',
    'Position': 'mean',
    'Requête': lambda x: ', '.join(x)
}).reset_index()

# Afficher le résumé des clusters
st.write("Résumé des clusters :")
st.dataframe(summary)

# Créer une visualisation
fig, ax = plt.subplots()
ax.scatter(summary['Clics'], summary['Impressions'], s=summary['Clics'] * 0.1, alpha=0.5)

# Annoter chaque point avec le nom du cluster
for i, row in summary.iterrows():
    ax.annotate(row['Nom du Cluster'], (row['Clics'], row['Impressions']), fontsize=8)

ax.set_xlabel('Clics')
ax.set_ylabel('Impressions')
ax.set_title('Visualisation des Clusters')
st.pyplot(fig)
