# 🌸 Projeto de Análise do Dataset Iris

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-latest-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completo-brightgreen.svg)]()

## 📋 Descrição

Este projeto implementa uma análise completa do famoso **Dataset Iris**, incluindo exploração de dados, visualizações, modelagem de machine learning e geração de relatórios. O dataset contém medições de características de três espécies de flores íris: *Setosa*, *Versicolor* e *Virginica*.

## 🎯 Objetivos

- 📊 Realizar análise exploratória dos dados
- 🔍 Visualizar correlações e padrões nos dados
- 🤖 Treinar modelo de classificação usando Regressão Logística
- 📈 Avaliar performance do modelo
- 📄 Gerar relatórios automáticos

## 🏗️ Estrutura do Projeto

```
├── data/
│   └── iris.csv                # Dataset principal
├── src/
│   ├── preprocessamento.py     # Carregamento e preparação dos dados
│   ├── visualizacao.py         # Funções de visualização
│   ├── modelo.py               # Criação e treinamento do modelo
│   ├── metricas.py             # Avaliação e métricas
│   └── relatorio.py            # Geração de relatórios
├── graficos/                   # Gráficos salvos automaticamente
├── main.py                     # Script principal
├── requirements.txt            # Dependências
├── relatorio.txt              # Relatório estatístico
└── README.md                  # Este arquivo
```

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **scikit-learn** - Machine Learning
- **matplotlib** - Visualizações básicas
- **seaborn** - Visualizações estatísticas
- **tensorflow** - Framework de ML (listado nas dependências)


```

## 📊 Funcionalidades

### 🔍 Análise Exploratória
- Carregamento automático do dataset
- Estatísticas descritivas completas
- Verificação de valores nulos
- Distribuição das classes

### 📈 Visualizações
- **Matriz de Correlação**: Mostra correlações entre as características
- **Gráficos de Dispersão**: 
  - Petal Length vs Petal Width
  - Sepal Length vs Sepal Width
- **Matriz de Confusão**: Avaliação visual do modelo

### 🤖 Modelagem
- Divisão treino/teste (80/20)
- Padronização dos dados usando StandardScaler
- Codificação de rótulos (LabelEncoder)
- Treinamento com Regressão Logística
- Avaliação com métricas de acurácia

### 📄 Relatórios
- Geração automática de relatório estatístico
- Salvamento de gráficos em PNG
- Organização em pastas estruturadas

## 📈 Resultados Esperados

O projeto gera automaticamente:

1. **Relatório Estatístico** (`relatorio.txt`):
   - Estatísticas descritivas completas
   - Contagem por espécie (50 amostras cada)
   - Verificação de integridade dos dados

2. **Gráficos** (pasta `graficos/`):
   - `correlacao.png` - Matriz de correlação
   - `dispersao_petalas.png` - Dispersão das pétalas
   - `dispersao_sepalas.png` - Dispersão das sépalas
   - `matriz_confusao.png` - Matriz de confusão

3. **Métricas do Modelo**:
   - Acurácia tipicamente > 95%
   - Classificação precisa das três espécies

## 🎯 Características do Dataset

| Característica | Descrição |
|---|---|
| **sepal.length** | Comprimento da sépala (cm) |
| **sepal.width** | Largura da sépala (cm) |
| **petal.length** | Comprimento da pétala (cm) |
| **petal.width** | Largura da pétala (cm) |
| **variety** | Espécie da íris |

### 📊 Distribuição das Classes
- **Setosa**: 50 amostras
- **Versicolor**: 50 amostras  
- **Virginica**: 50 amostras

**Total**: 150 amostras (dataset balanceado)

## 🔧 Configurações

### Parâmetros Principais
```python
# Divisão dos dados
test_size = 0.2
random_state = 42

# Modelo
LogisticRegression(max_iter=200)

# Visualizações
figsize = (8, 6)
palette = "Set2"
```


Desenvolvido com ❤️ para aprendizado de Machine Learning e análise de dados.


