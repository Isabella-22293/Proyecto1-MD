import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Cargar datos
data = pd.read_csv("train.csv")

# Crear variables predictoras con one-hot encoding y eliminar la variable respuesta
X = pd.get_dummies(data.drop(columns=['SalePrice']))
# Imputar valores faltantes en X reemplazándolos con la media de cada columna
X = X.fillna(X.mean())

y = data['SalePrice']

# Dividir en conjuntos de entrenamiento y validación (80% entrenamiento, 20% validación)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def learning_curve(model, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []
    m = len(X_train)
    for i in range(10, m, int(m/20)):
        model.fit(X_train[:i], y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_val_predict = model.predict(X_val)
        train_errors.append(np.sqrt(mean_squared_error(y_train[:i], y_train_predict)))
        val_errors.append(np.sqrt(mean_squared_error(y_val, y_val_predict)))
    plt.plot(train_errors, "r-+", linewidth=2, label="Error en Entrenamiento")
    plt.plot(val_errors, "b-", linewidth=3, label="Error en Validación")
    plt.xlabel("Tamaño del entrenamiento")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Curva de Aprendizaje")
    plt.show()

# modelo de Regresión Lineal:
model_lr = LinearRegression()
learning_curve(model_lr, X_train, y_train, X_val, y_val)
