import pandas as pd
import numpy as np

def feature_engineering(df):    
    print("Antes de modificar el dataset:", df.shape)

    # Verificar si las columnas necesarias existen
    required_columns = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath",
                        "BsmtFullBath", "BsmtHalfBath", "WoodDeckSF", "OpenPorchSF",
                        "EnclosedPorch", "3SsnPorch", "ScreenPorch", "YrSold", "YearBuilt",
                        "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                        "KitchenQual"]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Advertencia: No se encontraron las siguientes columnas en el dataset: {missing_cols}")
        return df  # Devolver sin modificaciones si faltan columnas clave

    # Área total combinada (sótano + pisos superiores)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df.get("2ndFlrSF", 0)

    # Número total de baños (baños completos y medios convertidos a 0.5)
    df["Bathrooms"] = (df["FullBath"] + df["HalfBath"] * 0.5 +
                       df["BsmtFullBath"] + df["BsmtHalfBath"] * 0.5)

    # Área total de porches
    df["PorchArea"] = (df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + 
                        df["3SsnPorch"] + df["ScreenPorch"])

    # Antigüedad de la casa
    df["Age"] = df["YrSold"] - df["YearBuilt"]

    # Años desde la última remodelación
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # Variables ordinales con una escala de calidad
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, np.nan: 0}
    for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual"]:
        df[col] = df[col].map(quality_map)

    # Variables nominales con One-Hot Encoding
    df = pd.get_dummies(df, columns=["Neighborhood", "HouseStyle", "RoofStyle", "Exterior1st"], drop_first=True)

    # Eliminar variables irrelevantes
    drop_cols = ["Id", "YrSold", "MoSold", "MiscFeature", "Alley", "PoolQC",
                 "LowQualFinSF", "BsmtFinSF2", "MiscVal"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    print("Después de modificar el dataset:", df.shape)
    
    return df
