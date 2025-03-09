import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from split_data import split_data

from tree_classification import preprocessing_for_classification

sns.set(style="whitegrid")
np.random.seed(42)

def main():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)
    
    df = preprocessing_for_classification(df)
    
    X = df.drop('PriceCategory', axis=1)
    y = df['PriceCategory']
    
    # Dividir el dataset en entrenamiento y prueba
    train_data, test_data = split_data(df)
    X_train = train_data.drop('PriceCategory', axis=1)
    y_train = train_data['PriceCategory']
    X_test = test_data.drop('PriceCategory', axis=1)
    y_test = test_data['PriceCategory']
    
    # Definir diferentes valores de profundidad para el árbol
    depths = [None, 3, 5, 7]
    results = []
    
    print("### Comparación de Árboles de Clasificación con Diferentes Profundidades ###")
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            'max_depth': d,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred)
        })
        print(f"\nmax_depth = {d}")
        print(f"  -> Exactitud: {acc:.4f}")
        print("  -> Reporte de Clasificación:")
        print(classification_report(y_test, y_pred))
    
    # Determinar el modelo con mayor exactitud
    best_model = max(results, key=lambda x: x['accuracy'])
    print("### Mejor Modelo Según Exactitud ###")
    print(f"max_depth = {best_model['max_depth']} con exactitud = {best_model['accuracy']:.4f}")
    
if __name__ == "__main__":
    main()
