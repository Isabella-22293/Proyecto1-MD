import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from tree_classification import preprocessing_for_classification

sns.set(style="whitegrid")
np.random.seed(42)

def main():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)
    
    df = preprocessing_for_classification(df)
    
    X = df.drop('PriceCategory', axis=1)
    y = df['PriceCategory']
    
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    # Configurar validación cruzada estratificada (5 folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluar el modelo usando cross_val_score (scoring='accuracy')
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    print("Exactitud en cada fold (validación cruzada, 5-fold):")
    print(cv_scores)
    print("Exactitud media (CV): {:.4f}".format(cv_scores.mean()))
    
    # Obtener predicciones a través de cross_val_predict para analizar la matriz de confusión y el reporte de clasificación
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)
    print("\nReporte de clasificación (validación cruzada):")
    print(classification_report(y, y_pred_cv))
    
    # Usar las clases únicas de y para la matriz de confusión
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred_cv, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión - Validación Cruzada")
    plt.xlabel("Categoría Predicha")
    plt.ylabel("Categoría Real")
    plt.show()
    
    # Análisis comparativo:
    print("\nAnálisis comparativo:")
    print("El modelo entrenado con simple split tenía una exactitud de aproximadamente 0.7705.")
    if cv_scores.mean() > 0.7705:
        print("La validación cruzada sugiere un desempeño ligeramente superior, con una exactitud media de CV de {:.4f}.".format(cv_scores.mean()))
    else:
        print("La validación cruzada muestra una exactitud media de CV de {:.4f}, que es similar o inferior al modelo anterior.".format(cv_scores.mean()))
    
if __name__ == "__main__":
    main()
