import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression

sns.set(style="whitegrid")

def compute_vif(X):
    """Calcula el VIF para cada variable en X."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def run_multivariate_regression(data_preprocessed):
    # Seleccionar todas las variables numéricas
    numeric_cols = data_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir la variable respuesta y la original del precio
    for col in ['SalePrice', 'SalePrice_log']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    # Variables independientes (X) y variable dependiente (y)
    X = data_preprocessed[numeric_cols]
    y = data_preprocessed['SalePrice_log']
    
    # Agregar la constante (intersección)
    X_sm = sm.add_constant(X)
    
    # Ajustar el modelo OLS
    model = sm.OLS(y, X_sm).fit()
    print(model.summary())
    
    # Calcular el VIF para detectar multicolinealidad
    vif = compute_vif(X_sm)
    print("\nVIF para las variables:")
    print(vif)
    
    # Predicciones y evaluación del modelo
    y_pred = model.predict(X_sm)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'\nMSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    
    # Gráfico: Predicciones vs Valores Reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel('Valores Reales (SalePrice_log)')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.show()
    
    # Análisis de residuos
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residuales")
    plt.title("Distribución de los Residuales")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicciones")
    plt.ylabel("Residuales")
    plt.title("Residuales vs Predicciones")
    plt.axhline(0, color='red', linestyle='--')
    plt.show()
    
    return model

def run_ridge_regression(data_preprocessed, alpha=1.0):
    """Aplica Ridge Regression para controlar el sobreajuste."""
    numeric_cols = data_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['SalePrice', 'SalePrice_log']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    X = data_preprocessed[numeric_cols]
    y = data_preprocessed['SalePrice_log']
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("Ridge Regression:")
    print(f"Alpha: {alpha}")
    print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.xlabel('Valores Reales (Test)')
    plt.ylabel('Predicciones (Test)')
    plt.title('Ridge: Predicciones vs Valores Reales (Test)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.show()

def main():
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    
    # Transformar la variable objetivo
    data['SalePrice_log'] = np.log(data['SalePrice'])
    
    # Imputar valores faltantes en variables numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    print("### Modelo de Regresión Lineal Multivariada ###")
    model = run_multivariate_regression(data)
    
    # Análisis de correlación: matriz de correlación
    plt.figure(figsize=(12, 10))
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.show()
    
    # Evaluar sobreajuste: comparar desempeño en entrenamiento y prueba
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['SalePrice', 'SalePrice_log']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    X = data[numeric_cols]
    y = data['SalePrice_log']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_train_pred = model_lr.predict(X_train)
    y_test_pred = model_lr.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("Evaluación con Linear Regression:")
    print("Entrenamiento: MSE = {:.4f}, R² = {:.4f}".format(train_mse, train_r2))
    print("Prueba: MSE = {:.4f}, R² = {:.4f}".format(test_mse, test_r2))
    
    # Si el desempeño en prueba es significativamente peor que en entrenamiento, se puede sospechar de sobreajuste.
    if test_r2 < train_r2 - 0.1:
        print("Se detecta posible sobreajuste. Se aplica Ridge Regression para regularizar el modelo.")
        run_ridge_regression(data, alpha=1.0)
    else:
        print("No se detecta sobreajuste significativo.")
    
if __name__ == "__main__":
    main()
