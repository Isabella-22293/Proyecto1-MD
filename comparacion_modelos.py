import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Convertir columnas categóricas a numéricas
def convertir_a_numerico(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Cargar datos
file_path = "train.csv"  
df = pd.read_csv(file_path)

# Selección de variables
X = df.drop(columns=['SalePrice'])  # Variables predictoras
y = df['SalePrice']  # Variable objetivo

# Convertir columnas categóricas a valores numéricos
X = convertir_a_numerico(X)

# Manejo de valores nulos solo en columnas numéricas
numerical_columns = X.select_dtypes(include=['number'])  # Filtrar solo las columnas numéricas
mean_values = numerical_columns.mean()  # Calcular la media de las columnas numéricas
X[numerical_columns.columns] = X[numerical_columns.columns].fillna(mean_values)  # Rellenar valores nulos

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Modelo Naïve Bayes (para regresión) ---------- #
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# ---------- Modelo Regresión Lineal ---------- #
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ---------- Modelo Árbol de Decisión ---------- #
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Función para calcular métricas
def evaluar_modelo(y_real, y_pred, nombre_modelo):
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    print(f"\n### Evaluación de {nombre_modelo} ###")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mse, rmse, mae, r2

# Evaluar los modelos
evaluar_modelo(y_test, y_pred_nb, "Naïve Bayes")
evaluar_modelo(y_test, y_pred_lr, "Regresión Lineal")
evaluar_modelo(y_test, y_pred_dt, "Árbol de Decisión")

# ---------- Gráficos ---------- #
def graficar_resultados(y_real, y_pred, titulo):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_real, y=y_pred)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title(titulo)
    plt.show()

graficar_resultados(y_test, y_pred_nb, "Predicción con Naïve Bayes")
graficar_resultados(y_test, y_pred_lr, "Predicción con Regresión Lineal")
graficar_resultados(y_test, y_pred_dt, "Predicción con Árbol de Decisión")
