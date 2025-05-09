import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def carregar_dados(caminho_csv):
    return pd.read_csv(caminho_csv)

def dividir_dados(df, test_size=0.2, random_state=42):
    X = df.drop('variety', axis=1)
    y = df['variety']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def padronizar(X_train, X_test):
    scaler=StandardScaler()
    X_train_pad=scaler.fit_transform(X_train)
    X_test_pad=scaler.transform(X_test)
    return X_train_pad, X_test_pad

