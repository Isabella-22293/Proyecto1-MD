import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def create_price_category(df):
    lower_threshold = df['SalePrice'].quantile(0.33)
    upper_threshold = df['SalePrice'].quantile(0.66)
    print(f"Umbral inferior (33%): {lower_threshold:.2f}")
    print(f"Umbral superior (66%): {upper_threshold:.2f}")
    
    def categorize(price):
        if price < lower_threshold:
            return "Económicas"
        elif price < upper_threshold:
            return "Intermedias"
        else:
            return "Caras"
    
    df['PriceCategory'] = df['SalePrice'].apply(categorize)
    return df

def plot_price_distribution(df):
    lower_threshold = df['SalePrice'].quantile(0.33)
    upper_threshold = df['SalePrice'].quantile(0.66)
    
    plt.figure(figsize=(10,6))
    sns.histplot(df['SalePrice'], kde=True, bins=30, color='skyblue')
    plt.axvline(lower_threshold, color='red', linestyle='--', label='33%')
    plt.axvline(upper_threshold, color='green', linestyle='--', label='66%')
    plt.title("Distribución de SalePrice con Umbrales de Categorías")
    plt.xlabel("SalePrice")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()

def main():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)
    
    df = create_price_category(df)
    
    # Mostrar la distribución de las categorías
    print("\nConteo de casas por categoría:")
    print(df['PriceCategory'].value_counts())
    
    # Graficar la distribución de SalePrice con los umbrales
    plot_price_distribution(df)

if __name__ == "__main__":
    main()
