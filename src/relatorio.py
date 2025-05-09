import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np 
from sklearn.metrics import confusion_matrix

def gerar_resumo_estatistico(df, caminho_saida="relatorio.txt"):
    """Gera um relatório estatístico em formato de texto"""
    with open(caminho_saida, "w") as f:
        f.write("📊 RESUMO ESTATÍSTICO DO DATASET\n\n")
        f.write(str(df.describe()) + "\n\n")
        f.write("📈 QUANTIDADE POR ESPÉCIE:\n")
        f.write(str(df['variety'].value_counts()) + "\n\n")
        f.write("📌 VERIFICAÇÃO DE VALORES NULOS:\n")
        f.write(str(df.isnull().sum()) + "\n")

def salvar_graficos(df, pasta="graficos"):
    """Salva os gráficos em arquivos PNG"""
    os.makedirs(pasta, exist_ok=True)


    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(f"{pasta}/correlacao.png")
    plt.close()


    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="petal.length", y="petal.width", hue="variety", palette="Set2", s=80)
    plt.title("Dispersão entre Petal Length e Petal Width")
    plt.tight_layout()
    plt.savefig(f"{pasta}/dispersao_petalas.png")
    plt.close()


    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="sepal.length", y="sepal.width", hue="variety", palette="Set2", s=80)
    plt.title("Dispersão entre Sepal Length e Sepal Width")
    plt.tight_layout()
    plt.savefig(f"{pasta}/dispersao_sepalas.png")
    plt.close()


def salvar_matriz_confusao(modelo, X_test, y_test, classes, pasta="graficos"):
    """Salva a matriz de confusão como imagem"""
    os.makedirs(pasta, exist_ok=True)

    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Previsão")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(f"{pasta}/matriz_confusao.png")
    plt.close()

