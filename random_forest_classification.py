import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tree_classification import preprocessing_for_classification
from split_data import split_data

sns.set(style="whitegrid")
np.random.seed(42)

def main():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)
    
    df = preprocessing_for_classification(df)
    
    X = df.drop('PriceCategory', axis=1)
    y = df['PriceCategory']
    
    train_data, test_data = split_data(df)
    X_train = train_data.drop('PriceCategory', axis=1)
    y_train = train_data['PriceCategory']
    X_test = test_data.drop('PriceCategory', axis=1)
    y_test = test_data['PriceCategory']
    
    # Entrenar el modelo Random Forest para clasificación
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = rf_clf.predict(X_test)
    
    # Evaluar el modelo
    acc = accuracy_score(y_test, y_pred)
    print("Exactitud del modelo Random Forest de clasificación: {:.4f}".format(acc))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Calcular y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=rf_clf.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=rf_clf.classes_, yticklabels=rf_clf.classes_)
    plt.title("Matriz de Confusión - Random Forest de Clasificación")
    plt.xlabel("Categoría Predicha")
    plt.ylabel("Categoría Real")
    plt.show()

if __name__ == "__main__":
    main()
