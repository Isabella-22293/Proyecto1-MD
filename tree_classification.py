# tree_classification.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from price_category import create_price_category
from split_data import split_data

sns.set(style="whitegrid")
np.random.seed(42)

def preprocessing_for_classification(df):
    df = create_price_category(df)
    
    for col in ['SalePrice', 'SalePrice_log']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Convertir a variables dummy las variables categóricas, excepto la respuesta 'PriceCategory'
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'PriceCategory' in cat_cols:
        cat_cols.remove('PriceCategory')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

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
    
    # Entrenar el árbol de clasificación (se usa un max_depth de 5, ajustable según requerimientos)
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    
    # Realizar predicciones y evaluar la eficiencia con el conjunto de prueba
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Exactitud del modelo de clasificación: {:.4f}".format(acc))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión - Árbol de Clasificación")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.show()
    
    # Visualizar gráficamente el árbol (limitado a los primeros 3 niveles para facilitar su interpretación)
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=X_train.columns, class_names=clf.classes_, filled=True, max_depth=3)
    plt.title("Árbol de Clasificación (Primeros 3 Niveles)")
    plt.show()

if __name__ == "__main__":
    main()
