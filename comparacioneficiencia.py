import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
data = pd.read_csv("train.csv")

# Preprocesamiento:
# - Convertir variables categóricas a numéricas con one-hot encoding.
# - Imputar valores faltantes con la media.
X = pd.get_dummies(data.drop(columns=['SalePrice']))
X = X.fillna(X.mean())

# Variable respuesta para clasificación (usaremos una discretización de SalePrice para este caso)
y = pd.qcut(data['SalePrice'], q=4, labels=False)  # Discretiza en 4 categorías

# División en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Árbol de Decisión
dt_classifier = DecisionTreeClassifier(random_state=42)
start_time = time.time()
dt_classifier.fit(X_train_scaled, y_train)
dt_time = time.time() - start_time
y_pred_dt = dt_classifier.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# Random Forest
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
start_time = time.time()
rf_classifier.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time
y_pred_rf = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)


print("Árbol de Decisión - Exactitud: {:.4f}, Tiempo de entrenamiento: {:.4f} segundos".format(dt_accuracy, dt_time))
print("Random Forest - Exactitud: {:.4f}, Tiempo de entrenamiento: {:.4f} segundos".format(rf_accuracy, rf_time))
