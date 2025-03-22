import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv("train.csv")

# Crear variable categórica
data['PriceCategory'] = pd.qcut(data['SalePrice'], q=3, labels=['Barata', 'Media', 'Cara'])

# Variables predictoras y respuesta
numeric_cols = data.select_dtypes(include=['number']).columns  # Solo columnas numéricas
X = data[numeric_cols]
y = data['PriceCategory']

# Imputar valores faltantes (NaN) con la media
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)  # Imputar las variables predictoras

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de Naïve Bayes
model_nb_class = MultinomialNB()
model_nb_class.fit(X_train_scaled, y_train)

# Predicción
y_pred_class = model_nb_class.predict(X_test_scaled)

# Evaluación: Exactitud y reporte de clasificación
print("### Evaluación del Modelo de Clasificación Naïve Bayes ###")
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Barata', 'Media', 'Cara'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Clasificación Naïve Bayes")
plt.show()
