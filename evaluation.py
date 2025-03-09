import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba y muestra las métricas:
    MSE, RMSE, MAE y R².
    """
    # Predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Mostrar resultados
    print("### Evaluación del Modelo en Conjunto de Prueba ###")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mse, rmse, mae, r2

# Ejemplo de integración en el flujo de trabajo
if __name__ == "__main__":
    # Supongamos que ya tienes el DataFrame 'data'
    # y la variable objetivo transformada: 'SalePrice_log'
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    # Seleccionar las variables predictoras (ejemplo, del modelo refinado)
    # Asegúrate de tener X y y definidos
    # Aquí usamos todas las variables numéricas como ejemplo:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['SalePrice', 'SalePrice_log']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    X = data[numeric_cols]
    y = data['SalePrice_log']
    
    # Dividir en entrenamiento y prueba (80%/20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # modelo de regresión lineal
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    evaluate_model(model_lr, X_test, y_test)
