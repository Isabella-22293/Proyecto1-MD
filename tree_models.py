import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importar funciones definidas en otros módulos del proyecto
from feature_engineering import feature_engineering
from split_data import split_data
from evaluation import evaluate_model

sns.set(style="whitegrid")
np.random.seed(42)

def create_price_category(df):
    """
    Crea la variable categórica 'PriceCategory' que agrupa los precios en tres categorías:
      - Económicas: precios menores al percentil 33.
      - Intermedias: precios entre el percentil 33 y el 66.
      - Caras: precios mayores al percentil 66.
    Se imprimen los umbrales para justificar la elección.
    """
    quantiles = df['SalePrice'].quantile([0.33, 0.66]).values
    low_threshold, high_threshold = quantiles[0], quantiles[1]
    print("Umbrales de clasificación de precios:")
    print(f"  Económicas: < {low_threshold:.2f}")
    print(f"  Intermedias: entre {low_threshold:.2f} y {high_threshold:.2f}")
    print(f"  Caras: > {high_threshold:.2f}")
    
    def categorize(price):
        if price < low_threshold:
            return "Económicas"
        elif price < high_threshold:
            return "Intermedias"
        else:
            return "Caras"
    
    df['PriceCategory'] = df['SalePrice'].apply(categorize)
    return df, low_threshold, high_threshold

def main():
    # Cargar el dataset (usando el mismo 'train.csv')
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    
    # Aplicar ingeniería de características (ya existente)
    data = feature_engineering(data)
    
    # Crear la nueva variable categórica para clasificación
    data, low_threshold, high_threshold = create_price_category(data)
    
    # Preprocesamiento básico: imputar valores faltantes y transformar SalePrice
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    data['SalePrice_log'] = np.log(data['SalePrice'])
    
    # Para el modelo de clasificación, codificamos la variable categórica en números
    le = LabelEncoder()
    data['PriceCategoryEncoded'] = le.fit_transform(data['PriceCategory'])
    
    # Dividir el dataset en entrenamiento y prueba usando la función split_data
    train_data, test_data = split_data(data)
    
    # Seleccionar las características para los modelos:
    # Se utilizan las columnas numéricas excluyendo la variable respuesta y las variables derivadas.
    features = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['SalePrice', 'SalePrice_log', 'PriceCategoryEncoded']:
        if col in features:
            features.remove(col)
    
    # --- MODELOS DE REGRESIÓN ---
    # Variables para regresión: predecir SalePrice_log
    X_train_reg = train_data[features]
    y_train_reg = train_data['SalePrice_log']
    X_test_reg = test_data[features]
    y_test_reg = test_data['SalePrice_log']
    
    # 1. Regresión Lineal (ya usado en la entrega anterior)
    model_lr = LinearRegression()
    model_lr.fit(X_train_reg, y_train_reg)
    print("\n### Evaluación de Regresión Lineal ###")
    mse_lr, rmse_lr, mae_lr, r2_lr = evaluate_model(model_lr, X_test_reg, y_test_reg)
    
    # 2. Árbol de Decisión para Regresión
    dt_reg = DecisionTreeRegressor(random_state=42)
    dt_reg.fit(X_train_reg, y_train_reg)
    y_pred_dt = dt_reg.predict(X_test_reg)
    mse_dt = mean_squared_error(y_test_reg, y_pred_dt)
    rmse_dt = np.sqrt(mse_dt)
    r2_dt = r2_score(y_test_reg, y_pred_dt)
    print("\n### Evaluación del Árbol de Decisión para Regresión ###")
    print(f"MSE: {mse_dt:.4f}")
    print(f"RMSE: {rmse_dt:.4f}")
    print(f"R²: {r2_dt:.4f}")
    
    # 3. Random Forest para Regresión
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_rf = rf_reg.predict(X_test_reg)
    mse_rf = mean_squared_error(y_test_reg, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test_reg, y_pred_rf)
    print("\n### Evaluación del Random Forest para Regresión ###")
    print(f"MSE: {mse_rf:.4f}")
    print(f"RMSE: {rmse_rf:.4f}")
    print(f"R²: {r2_rf:.4f}")
    
    # --- MODELO DE CLASIFICACIÓN ---
    # Variables para clasificación: predecir PriceCategoryEncoded
    X_train_clf = train_data[features]
    y_train_clf = train_data['PriceCategoryEncoded']
    X_test_clf = test_data[features]
    y_test_clf = test_data['PriceCategoryEncoded']
    
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train_clf, y_train_clf)
    y_pred_clf = dt_clf.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    print("\n### Evaluación del Árbol de Decisión para Clasificación ###")
    print(f"Exactitud: {acc:.4f}")
    print("Reporte de Clasificación:")
    print(classification_report(y_test_clf, y_pred_clf, target_names=le.classes_))
    
    # --- VISUALIZACIÓN DE LOS ÁRBOLES (opcional) ---
    plt.figure(figsize=(20,10))
    plot_tree(dt_reg, feature_names=features, filled=True, max_depth=3)
    plt.title("Árbol de Decisión para Regresión (Primeros 3 Niveles)")
    plt.show()
    
    plt.figure(figsize=(20,10))
    plot_tree(dt_clf, feature_names=features, filled=True, max_depth=3)
    plt.title("Árbol de Decisión para Clasificación (Primeros 3 Niveles)")
    plt.show()

if __name__ == "__main__":
    main()
