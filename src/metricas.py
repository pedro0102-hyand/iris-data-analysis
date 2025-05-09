import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

def plotar_matriz_confusao(modelo, X_test, y_test, classes):
    y_pred = modelo.predict(X_test)
    cm=confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Previsao")
    plt.ylabel("Real")
    plt.title("Matriz de confusao")
    plt.tight_layout()
    plt.show()