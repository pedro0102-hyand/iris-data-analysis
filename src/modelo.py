from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def codificar_rotulos(y_train, y_test):
    """Transforma rótulos em valores numéricos (ex: Setosa → 0)"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    return y_train_encoded, y_test_encoded, le.classes_

def treinar_modelo(X_train, y_train):
    """Cria e treina um modelo simples"""
    modelo = LogisticRegression(max_iter=200)  
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo e retorna acurácia"""
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


