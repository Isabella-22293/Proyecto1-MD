import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from price_category import create_price_category
from split_data import split_data

sns.set(style="whitegrid")
np.random.seed(42)

def preprocess_for_confusion_analysis(df):
    df = create_price_category(df)
    for col in ['SalePrice', 'SalePrice_log']:
        if col in df.columns:
            df = df.drop(columns=[col])
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'PriceCategory' in cat_cols:
        cat_cols.remove('PriceCategory')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

def main():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)    
    df = preprocess_for_confusion_analysis(df)
    
    train_data, test_data = split_data(df)
    X_train = train_data.drop('PriceCategory', axis=1)
    y_train = train_data['PriceCategory']
    X_test = test_data.drop('PriceCategory', axis=1)
    y_test = test_data['PriceCategory']
    
    # Entrenar el árbol de clasificación (usando max_depth=5, ajustable)
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = clf.predict(X_test)
    
    # Calcular y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("Matriz de Confusión:")
    print(cm)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Matriz de Confusión - Árbol de Clasificación")
    plt.xlabel("Categoría Predicha")
    plt.ylabel("Categoría Real")
    plt.show()
    
    # Mostrar el reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=clf.classes_)
    print("Reporte de Clasificación:")
    print(report)
    
    # Análisis de la eficiencia basado en la matriz de confusión
    classes = list(clf.classes_)
    total_instances = np.sum(cm)
    total_correct = np.trace(cm)
    total_errors = total_instances - total_correct
    print(f"\nTotal de instancias evaluadas: {total_instances}")
    print(f"Instancias correctamente clasificadas: {total_correct}")
    print(f"Instancias mal clasificadas: {total_errors}\n")
    
    print("Análisis por categoría:")
    for i, actual in enumerate(classes):
        total_actual = np.sum(cm[i, :])
        correct = cm[i, i]
        errors = total_actual - correct
        error_rate = errors / total_actual if total_actual > 0 else 0
        print(f"  - {actual}:")
        print(f"       Total: {total_actual}")
        print(f"       Correctos: {correct}")
        print(f"       Errores: {errors} (Tasa de error: {error_rate:.2f})")
    
    # Comentario general sobre la importancia de los errores:
    print("\nInterpretación:")
    print(" - Las categorías con mayor número o tasa de error indican dónde el modelo tiene más dificultad para distinguir las clases.")
    print(" - Por ejemplo, si 'Intermedias' tiene una tasa de error elevada, puede deberse a que sus características se solapan con las de 'Económicas' o 'Caras'.")
    print(" - Los errores en ciertas categorías pueden ser más críticos dependiendo del contexto; por ejemplo, confundir una casa 'Cara' con una 'Económica' puede tener implicaciones importantes en la toma de decisiones.")
    
if __name__ == "__main__":
    main()
