# Fundamentos de Machine Learning — USP Hype Data & AI

Portfólio acadêmico com exercícios desenvolvidos durante o curso de extensão **Introdução a Machine Learning — Hype Data & AI, da Universidade de São Paulo (USP)**.

O repositório reúne exemplos comentados de aprendizagem supervisionada e não supervisionada, passando por coleta de dados, classificação, validação cruzada, otimização de hiperparâmetros, avaliação de modelos, análise de overfitting e clusterização.

> **Observação:** este é um material autoral de estudo e não representa um repositório oficial do curso ou da USP.

## Objetivos

- compreender diferenças entre aprendizagem supervisionada e não supervisionada;
- preparar variáveis para problemas de classificação;
- treinar e otimizar modelos de classificação;
- avaliar generalização com validação cruzada;
- interpretar precisão, recall e F1-score;
- investigar overfitting;
- analisar importância de atributos;
- aplicar K-Means e o método do cotovelo.

## Trilhas práticas

### 1. Classificação com Random Forest

O exemplo utiliza o dataset **Wine**, incluído no Scikit-learn, para classificar três categorias de vinho a partir de atributos físico-químicos.

Fluxo implementado:

1. carregamento do dataset;
2. divisão estratificada em 70% para treino e 30% para teste;
3. definição de uma grade de hiperparâmetros;
4. busca com `GridSearchCV` e validação cruzada de cinco folds;
5. seleção pelo F1-score macro;
6. avaliação no conjunto de teste;
7. análise da importância das características;
8. comparação entre desempenho de treino e validação.

Arquivos:

- [`random_forest.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/random_forest.py)
- [`métricas_de_avaliação_do_modelo.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/métricas_de_avaliação_do_modelo.py)
- [`importâncias_das_features_do_modelo.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/importâncias_das_features_do_modelo.py)
- [`verificando_overfitting.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/verificando_overfitting.py)

### 2. Validação cruzada com KNN

O script utiliza o dataset **Iris** para comparar os valores de `k` 3, 5, 7, 9, 13 e 17.

A avaliação combina:

- divisão de 80% para treino e 20% para teste;
- validação cruzada com dez folds;
- F1-score macro de treino e validação;
- F1-score final no conjunto de teste.

Arquivo:

- [`validação_cruzada_com_knn.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/validação_cruzada_com_knn.py)

### 3. Clusterização com K-Means

A trilha não supervisionada calcula o **WCSS** para valores de `k` entre 1 e 10, utiliza o método do cotovelo como apoio à escolha do número de grupos e apresenta um exemplo de treinamento com quatro clusters.

Arquivos:

- [`número_ideal_de_clusters_com_o_método_do_cotovelo.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/número_ideal_de_clusters_com_o_método_do_cotovelo.py)
- [`treinamento_do_k_means_com_k=4.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/treinamento_do_k_means_com_k=4.py)

### 4. Coleta e preparação do Adult Census

O exercício lê o dataset **Adult** diretamente do UCI Machine Learning Repository, atribui nomes às colunas e cria a variável binária `income_binary`:

- `0`: renda anual de até US$ 50 mil;
- `1`: renda anual superior a US$ 50 mil.

Arquivo:

- [`coleta_de_dados_dataset.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/coleta_de_dados_dataset.py)

### 5. Conceitos fundamentais

O material introdutório diferencia problemas supervisionados e não supervisionados por meio de exemplos de classificação, regressão e agrupamento.

Arquivos:

- [`aprendizado_supervisionado_vs_não_supervisionado.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/aprendizado_supervisionado_vs_não_supervisionado.py)
- [`avaliação_da_solução.py`](Curso%20de%20extensão%20introdução%20Machine%20Learning%20-%20Hype%20data%20%26%20Ai%20-USP/avaliação_da_solução.py)

## Pipeline de modelagem praticado

```text
Definição do problema
        │
        ▼
Coleta e preparação dos dados
        │
        ▼
Divisão entre treino e teste
        │
        ▼
Treinamento e validação cruzada
        │
        ▼
Otimização de hiperparâmetros
        │
        ▼
Avaliação e interpretação
```

Na aprendizagem não supervisionada, a etapa de avaliação é adaptada para comparar a compactação dos clusters por WCSS e inspecionar visualmente os grupos resultantes.

## Tecnologias utilizadas

- **Python 3**
- **Pandas** — leitura e preparação de dados;
- **Scikit-learn** — datasets, modelos, validação e métricas;
- **Matplotlib** — visualização do método do cotovelo e dos clusters;
- **Google Colab** — ambiente original dos exercícios.

## Conceitos e técnicas demonstrados

| Categoria | Conteúdos |
|---|---|
| Preparação | leitura por URL, definição de colunas e criação de variável-alvo |
| Classificação | Random Forest e K-Nearest Neighbors |
| Clusterização | K-Means e inicialização k-means++ |
| Validação | holdout, estratificação, cross-validation de 5 e 10 folds |
| Otimização | Grid Search sobre hiperparâmetros |
| Métricas | precisão, recall, relatório de classificação e F1-score macro |
| Diagnóstico | comparação treino × validação e verificação de overfitting |
| Interpretação | importância das características do Random Forest |
| Visualização | curva do cotovelo, clusters e centroides |

## Como executar

1. Clone o repositório:

```bash
git clone https://github.com/pablosena28/machine-learning-usp-hype.git
cd machine-learning-usp-hype
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install pandas scikit-learn matplotlib
```

4. Execute um script independente, por exemplo:

```bash
python "Curso de extensão introdução Machine Learning - Hype data & Ai -USP/random_forest.py"
```

## Ordem de execução e dependências

Alguns arquivos foram exportados de células do Google Colab e dependem de objetos criados em etapas anteriores.

### Random Forest

Execute primeiro `random_forest.py`. Na mesma sessão interativa, os scripts de métricas, importância e overfitting utilizam objetos como `grid_search`, `data`, `X_test` e `y_test`.

### K-Means

Os dois scripts de clusterização esperam que uma matriz numérica chamada `X` já esteja carregada e preparada. A versão atual não define essa fonte dentro dos próprios arquivos.

Por essa razão, os scripts independentes podem ser executados diretamente, enquanto as etapas dependentes funcionam melhor quando reunidas em um notebook ou módulo único.

## Estrutura atual

```text
.
├── README.md
└── Curso de extensão introdução Machine Learning - Hype data & Ai -USP/
    ├── aprendizado_supervisionado_vs_não_supervisionado.py
    ├── avaliação_da_solução.py
    ├── coleta_de_dados_dataset.py
    ├── coleta_de_dados_dataset (1).py
    ├── importâncias_das_features_do_modelo.py
    ├── métricas_de_avaliação_do_modelo.py
    ├── número_ideal_de_clusters_com_o_método_do_cotovelo.py
    ├── random_forest.py
    ├── treinamento_do_k_means_com_k=4.py
    ├── validação_cruzada_com_knn.py
    └── verificando_overfitting.py
```

## Limitações

- parte dos scripts depende do estado de uma sessão anterior do Colab;
- os resultados de execução não estão persistidos no repositório;
- os exemplos de K-Means não incluem a preparação da matriz `X`;
- há duas cópias idênticas do script de coleta do Adult Census;
- não há arquivo de dependências com versões fixadas;
- os scripts ainda não possuem testes automatizados;
- datasets externos podem mudar de endereço ou ficar temporariamente indisponíveis.

## Próximas melhorias

- transformar cada trilha em um notebook reproduzível;
- remover o arquivo duplicado de coleta;
- tornar cada script executável de forma independente;
- adicionar `requirements.txt` ou `pyproject.toml`;
- salvar tabelas de métricas e gráficos gerados;
- incluir matriz de confusão e curvas de aprendizado;
- adicionar padronização das variáveis antes do KNN e do K-Means;
- documentar a fonte utilizada na clusterização;
- criar testes para preparação de dados e métricas;
- comparar modelos sob o mesmo protocolo de validação.

## Autor

**Pablo Sena**

Portfólio de estudos em machine learning, análise de dados e modelagem preditiva.

[GitHub](https://github.com/pablosena28)
