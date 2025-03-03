import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")
np.random.seed(42)

# Variables para la clusterización
def prepare_cluster_data(data):
    features = ['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF']
    # Imputar valores faltantes con la mediana para estas variables
    data_cluster = data[features].fillna(data[features].median())
    return data_cluster

# Estandarización de los datos
def scale_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# K-means para clusterización
def apply_kmeans(data_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    return clusters, kmeans

# Reducción de dimensionalidad con PCA para visualización en 2D
def reduce_dimensionality(data_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data_scaled)
    return data_pca, pca

# Visualización de clusters en 2D
def plot_clusters_2d(data_pca, clusters):
    pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', alpha=0.7)
    plt.title('Clusters de viviendas (PCA 2D)')
    plt.show()

def cluster_analysis(data):
    data_cluster = prepare_cluster_data(data)
    data_scaled = scale_data(data_cluster)
    clusters, kmeans_model = apply_kmeans(data_scaled, n_clusters=3)
    data_cluster['Cluster'] = clusters
    print("Medias por Cluster:")
    print(data_cluster.groupby('Cluster').mean())
    data_pca, pca_model = reduce_dimensionality(data_scaled, n_components=2)
    plot_clusters_2d(data_pca, clusters)

def main():
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    print("### Análisis Exploratorio Previo ###")    
    print("\n### Análisis de Grupos (Clusterización) ###")
    cluster_analysis(data)

if __name__ == "__main__":
    main()
