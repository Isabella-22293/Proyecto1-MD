import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
data = pd.read_csv("train.csv")

# Preprocesamiento para regresión
X = pd.get_dummies(data.drop(columns=['SalePrice']))
X = X.fillna(X.mean())

y = data['SalePrice']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- GridSearch para Árbol de Decisión (Regresión) ---
param_grid_dt = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

dt = DecisionTreeRegressor(random_state=42)
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)

print("Mejores hiperparámetros (Árbol de Decisión):", grid_search_dt.best_params_)
print("Mejor RMSE:", np.sqrt(-grid_search_dt.best_score_))

# --- GridSearch para Naïve Bayes (Clasificación) ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir 'y_train' en categorías si es necesario
y_train_class = pd.qcut(y_train, q=5, labels=False)  # Dividir en 5 clases basadas en cuantiles

# Verificar balance de clases
print("Distribución de clases en y_train_class:")
print(pd.Series(y_train_class).value_counts())

param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_nb_class = MultinomialNB()
grid_search_nb = GridSearchCV(model_nb_class, param_grid_nb, cv=stratified_cv, scoring='accuracy')
grid_search_nb.fit(X_train_scaled, y_train_class)

print("Mejor hiperparámetro para Naïve Bayes:", grid_search_nb.best_params_)
print("Mejor exactitud:", grid_search_nb.best_score_)
