import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from feature_engineering import feature_engineering
from split_data import split_data

sns.set(style="whitegrid")
np.random.seed(42)

def preprocessing_all_vars(data):
    data = feature_engineering(data)
    numeric_cols = data.select_dtypes(include=['float64','int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    data = pd.get_dummies(data)
    return data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

def main():
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    
    # Preprocesamiento 
    data_preprocessed = preprocessing_all_vars(data)
    if 'SalePrice' not in data_preprocessed.columns:
        print("Error: 'SalePrice' no se encuentra en el dataset preprocesado.")
        return
    
    #Separar características y variable objetivo
    X = data_preprocessed.drop('SalePrice', axis=1)
    y = data_preprocessed['SalePrice']
    
    #Dividir en entrenamiento y prueba (80% / 20%)
    train_data, test_data = split_data(data_preprocessed)
    X_train = train_data.drop('SalePrice', axis=1)
    y_train = train_data['SalePrice']
    X_test = test_data.drop('SalePrice', axis=1)
    y_test = test_data['SalePrice']
    
    #Definir profundidades que queremos comparar
    depths = [None, 3, 5, 7]  # None = sin límite de profundidad
    
    #Entrenar y evaluar cada árbol
    results = []
    for d in depths:
        model = DecisionTreeRegressor(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        
        mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        
        results.append({
            'max_depth': d,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
    
    #Mostrar resultados
    print("\n### Comparación de Árboles de Decisión con Distintas Profundidades ###")
    for res in results:
        print(f"max_depth = {res['max_depth']}")
        print(f"  -> MSE:  {res['MSE']:.2f}")
        print(f"  -> RMSE: {res['RMSE']:.2f}")
        print(f"  -> MAE:  {res['MAE']:.2f}")
        print(f"  -> R²:   {res['R2']:.4f}\n")
    
    #Determinar el mejor modelo 
    #R² para elegir el mejor.
    best_model = max(results, key=lambda x: x['R2'])
    print("### Mejor Modelo según R² ###")
    print(f"max_depth = {best_model['max_depth']} con R² = {best_model['R2']:.4f}")

if __name__ == "__main__":
    main()
