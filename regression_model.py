import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

def backward_elimination(X, y, significance_level=0.05):
    """
    Proceso de eliminación hacia atrás:
      - Se ajusta un modelo OLS con todas las variables de X.
      - Se eliminan iterativamente las variables con p-valor mayor al umbral.
    Retorna la lista de variables seleccionadas y el modelo final.
    """
    features = list(X.columns)
    while len(features) > 0:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        p_values = model.pvalues.drop("const")
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            worst_feature = p_values.idxmax()
            features.remove(worst_feature)
            print(f"Eliminando {worst_feature} (p-value = {max_p_value:.4f})")
        else:
            break
    return features, model

def run_refined_model(data_preprocessed):
    """
    Selecciona, a partir de todas las variables numéricas (excluyendo la respuesta),
    las mejores predictoras del precio (transformado logarítmicamente) mediante backward elimination.
    Luego, muestra el resumen del modelo y analiza gráficamente los residuos.
    """
    # Seleccionar variables numéricas y excluir 'SalePrice' y 'SalePrice_log'
    numeric_cols = data_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['SalePrice', 'SalePrice_log']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    X = data_preprocessed[numeric_cols]
    y = data_preprocessed['SalePrice_log']
    
    # Aplicar backward elimination para seleccionar las variables significativas
    selected_features, refined_model = backward_elimination(X, y, significance_level=0.05)
    print("\nVariables seleccionadas:", selected_features)
    print(refined_model.summary())
    
    # Predicciones y análisis de residuos del modelo refinado
    X_selected = sm.add_constant(X[selected_features])
    y_pred = refined_model.predict(X_selected)
    residuals = y - y_pred

    # Gráfico: Valores Ajustados vs Residuales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Residuales")
    plt.title("Residuales vs Valores Ajustados (Modelo Refinado)")
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

    # Gráfico: Distribución de los Residuales
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residuales")
    plt.title("Distribución de los Residuales (Modelo Refinado)")
    plt.show()

def main():
    file_path = 'train.csv'
    data = pd.read_csv(file_path)
    
    # Transformar la variable respuesta para mejorar la linealidad
    data['SalePrice_log'] = np.log(data['SalePrice'])
    
    # Imputar valores faltantes en las variables numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    print("### Modelo Refinado con las Mejores Variables Predictoras ###")
    run_refined_model(data)

if __name__ == "__main__":
    main()
