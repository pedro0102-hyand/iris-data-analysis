from src import preprocessamento
from src import visualizacao
from src import modelo
from src import metricas 
from src import relatorio

def main():
   
    caminho = "data/iris.csv"
    df = preprocessamento.carregar_dados(caminho)

    print("✅ Dados carregados:")
    print(df.head())

    
    visualizacao.correlacao(df)
    visualizacao.dispersao(df, "petal.length", "petal.width")
    visualizacao.dispersao(df, "sepal.length", "sepal.width")

   
    X_train, X_test, y_train, y_test = preprocessamento.dividir_dados(df)
    print(f"\n✅ Formato dos dados:")
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

   
    X_train_pad, X_test_pad = preprocessamento.padronizar(X_train, X_test)
    print(f"\n✅ Dados padronizados. Média de X_train_pad[0]: {X_train_pad[0].mean():.2f}")

    
    y_train_encoded, y_test_encoded, classes = modelo.codificar_rotulos(y_train, y_test)
    print(f"\n✅ Classes detectadas: {list(classes)}")

    modelo_sklearn = modelo.treinar_modelo(X_train_pad, y_train_encoded)
    print("\n🚀 Modelo treinado com sucesso!")

    
    acc = modelo.avaliar_modelo(modelo_sklearn, X_test_pad, y_test_encoded)
    print(f"\n🎯 Acurácia no conjunto de teste: {acc:.2f}")

    metricas.plotar_matriz_confusao(modelo_sklearn, X_test_pad, y_test_encoded, classes)

    relatorio.gerar_resumo_estatistico(df)
    relatorio.salvar_graficos(df)
    print("\n📁 Relatório e gráficos salvos nas pastas!")

if __name__ == "__main__":
    main()


