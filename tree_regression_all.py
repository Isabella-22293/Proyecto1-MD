import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Importar funciones definidas en otros módulos del proyecto
from feature_engineering import feature_engineering
from split_data import split_data

sns.set(style="whitegrid")
np.random.seed(42)

def preprocessing_all_vars(data):
    """
    Preprocesamiento para utilizar todas las variables:
      - Se aplica la ingeniería de características.
      - Se imputan los valores faltantes para las variables numéricas.
      - Se convierten las variables categóricas a variables dummy.
    """
    # Aplicar feature engineering
    data = feature_engineering(data)
    
    # Imputar valores faltantes en variables numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # Convertir todas las variables categóricas en variables dummy
    data_preprocessed = pd.get_dummies(data)
    
    return data_preprocessed

def evaluate_regression(model, X_test, y_test):
    """
    Evalúa el modelo de regresión e imprime las métricas: MSE, RMSE, MAE y R².
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("### Evaluación del Árbol de Regresión con Todas las Variables ###")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mse, rmse, mae, r2

def main():
    # Cargar el dataset (se asume que el archivo 'train.csv' está en la raíz del proyecto)
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    
    # Preprocesamiento: aplicar ingeniería de características y conversión de variables categóricas
    data_preprocessed = preprocessing_all_vars(data)
    
    # Se asume que la variable objetivo es 'SalePrice' (sin transformar)
    if 'SalePrice' not in data_preprocessed.columns:
        print("Error: 'SalePrice' no se encuentra en el dataset preprocesado.")
        return
    
    # Separar las características y la variable objetivo
    X = data_preprocessed.drop('SalePrice', axis=1)
    y = data_preprocessed['SalePrice']
    
    # Dividir en conjuntos de entrenamiento y prueba usando la función split_data
    train_data, test_data = split_data(data_preprocessed)
    X_train = train_data.drop('SalePrice', axis=1)
    y_train = train_data['SalePrice']
    X_test = test_data.drop('SalePrice', axis=1)
    y_test = test_data['SalePrice']
    
    # Entrenar el árbol de regresión usando todas las variables
    dt_reg_all = DecisionTreeRegressor(random_state=42)
    dt_reg_all.fit(X_train, y_train)
    
    # Evaluar el modelo
    evaluate_regression(dt_reg_all, X_test, y_test)
    
    # Visualizar el árbol (limitado a los primeros 3 niveles para claridad)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_reg_all, feature_names=X_train.columns, filled=True, max_depth=3)
    plt.title("Árbol de Regresión con Todas las Variables (Primeros 3 Niveles)")
    plt.show()

if __name__ == "__main__":
    main()
