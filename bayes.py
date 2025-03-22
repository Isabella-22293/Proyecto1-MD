import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Cargar datos
file_path = "train.csv"
data = pd.read_csv(file_path)

# Selección de variables numéricas
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
target = 'SalePrice'
if target in numeric_cols:
    numeric_cols.remove(target)

X = data[numeric_cols]
y = data[target]

# Manejo de valores NaN - Imputación con la media de cada columna
X.fillna(X.mean(), inplace=True)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Naïve Bayes para regresión (Gaussian Naïve Bayes)
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# Predicción
y_pred = model_nb.predict(X_test)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicción de precios con Naïve Bayes")
plt.show()


# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
print("### Evaluación del Modelo de Regresión Naïve Bayes ###")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")