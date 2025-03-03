import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from split_data import split_data, compare_distribution  
from feature_engineering import feature_engineering

# Estilos de gráficos y reproducibilidad
sns.set(style="whitegrid")
np.random.seed(42)

# Cargar el dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Revisar el dataset
def data_overview(data):
    print("Dimensiones del dataset:", data.shape)
    print("\nTipos de variables:\n", data.dtypes)
    print("\nResumen estadístico:\n", data.describe())
    
    missing_values = data.isnull().sum().sort_values(ascending=False)
    print("\nDatos faltantes por variable:\n", missing_values[missing_values > 0])

# Visualizaciones
def plot_histograms(data):
    #Histogramas y diagramas de densidad para las variables clave
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(data['SalePrice'], kde=True)
    plt.title('Distribución de SalePrice')

    plt.subplot(1, 2, 2)
    sns.histplot(data['GrLivArea'], kde=True)
    plt.title('Distribución de GrLivArea')

    plt.tight_layout()
    plt.show()

def plot_boxplot(data):
    #Boxplot para detectar outliers
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data['SalePrice'])
    plt.title('Boxplot de SalePrice')
    plt.show()

def plot_correlation_matrix(data):
    # Seleccionar únicamente las columnas numéricas
    numeric_data = data.select_dtypes(include=[np.number])
    # Calcular la matriz de correlación
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title('Matriz de Correlación')
    plt.show()


def plot_scatter(data):
    #Diagrama de dispersión entre GrLivArea y SalePrice
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=data)
    plt.title('GrLivArea vs SalePrice')
    plt.show()

# Preprocesamiento
def preprocessing(data):
    # Imputar valores faltantes para variables numéricas
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    
    # Transformar SalePrice para normalizar su distribución
    data['SalePrice_log'] = np.log(data['SalePrice'])
    
    # Convertir variables categóricas a variables dummy 
    data = pd.get_dummies(data)
    
    return data

def main():
    file_path = 'train.csv'
    data = load_data(file_path)

    print("### Visión General de los Datos ###")
    data_overview(data)

    print("\n### Aplicando Ingeniería de Características ###")
    data = feature_engineering(data)
    
    print("\n### Visualizaciones ###")
    plot_histograms(data)
    plot_boxplot(data)
    plot_correlation_matrix(data)
    plot_scatter(data)

    print("\n### Preprocesamiento de los Datos ###")
    data_preprocessed = preprocessing(data)
    print("Preprocesamiento completado. Nuevas dimensiones:", data_preprocessed.shape)

    print("\n### División del Conjunto de Datos ###")
    train_data, test_data = split_data(data_preprocessed)
    compare_distribution(train_data, test_data)
    
if __name__ == "__main__":
    main()