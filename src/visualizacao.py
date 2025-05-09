import seaborn as sns
import matplotlib.pyplot as plt 

def correlacao(df):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True),annot=True, cmap="coolwarm",fmt=".2f")
    plt.title("Matriz de Correlacao")
    plt.tight_layout()
    plt.show()

def dispersao(df, x, y , hue="variety"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="Set2",s=80)
    plt.title(f"Dispersao entre {x} e {y}")
    plt.tight_layout()
    plt.show()
