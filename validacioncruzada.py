import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv("train.csv")

# Variables predictoras: convertir variables categóricas a numéricas y tratar valores faltantes
X = pd.get_dummies(data.drop(columns=['SalePrice']))
X = X.fillna(X.mean())

# Variable respuesta para regresión
y_reg = data['SalePrice']

# Variable respuesta para clasificación: se crea una variable categórica a partir de SalePrice
y_class = pd.qcut(data['SalePrice'], q=3, labels=['Barata', 'Media', 'Cara'])

# ------------------- MODELO DE REGRESIÓN -------------------

# Crear el modelo de Regresión Lineal
model_lr = LinearRegression()

# Validación cruzada: usamos 10 particiones (folds) y el scoring negativo de MSE
cv_scores = cross_val_score(model_lr, X, y_reg, cv=10, scoring='neg_mean_squared_error')

# Convertir las puntuaciones a RMSE
rmse_scores = np.sqrt(-cv_scores)
print("RMSE promedio (validación cruzada) - Regresión:", rmse_scores.mean())

# ------------------- MODELO DE CLASIFICACIÓN -------------------

# Escalar las características para el modelo de Naïve Bayes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Crear el modelo de Naïve Bayes (MultinomialNB)
model_nb_class = MultinomialNB()

# Validación cruzada: usamos 10 particiones (folds) para clasificación con scoring de exactitud
cv_scores_class = cross_val_score(model_nb_class, X_scaled, y_class, cv=10, scoring='accuracy')
print("Exactitud promedio (validación cruzada) - Clasificación:", cv_scores_class.mean())
