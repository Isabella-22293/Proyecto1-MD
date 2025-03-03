import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def split_data(data):
    #Divide el dataset en un 80% de entrenamiento y un 20% de prueba.
    #Se fija random_state para garantizar reproducibilidad.
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print("Filas en entrenamiento:", train_data.shape[0])
    print("Filas en prueba:", test_data.shape[0])
    return train_data, test_data

def compare_distribution(train_data, test_data):
    #Compara la distribución de la variable 'SalePrice' en los conjuntos de entrenamiento y prueba.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(train_data['SalePrice'], label="Entrenamiento", fill=True)
    sns.kdeplot(test_data['SalePrice'], label="Prueba", fill=True)
    plt.title("Comparación de Distribución de SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()
