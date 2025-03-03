import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_regression(data_preprocessed):
    X = data_preprocessed[['GrLivArea']]
    y = data_preprocessed['SalePrice_log']

    # Modelo statsmodels
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    print(model_sm.summary())

    # Modelo sklearn
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Evaluación del modelo
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, alpha=0.5, label="Datos reales")
    plt.plot(X, y_pred, color='red', linewidth=2, label="Regresión lineal")
    plt.xlabel("GrLivArea")
    plt.ylabel("SalePrice (log)")
    plt.title("Regresión Lineal Univariada")
    plt.legend()
    plt.show()

    # Residuales
    residuals = y - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residuales")
    plt.title("Distribución de los Residuales")
    plt.show()
    
def main():
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    data['SalePrice_log'] = np.log(data['SalePrice'])
    print("### Regression Model ###") 
    run_regression(data)
    
if __name__ == "__main__":
    main()
