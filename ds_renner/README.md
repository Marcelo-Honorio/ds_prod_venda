# **Previsão de Demanda de Produtos no Varejo**

## **Objetivo**
Este projeto tem como objetivo prever a demanda de produtos para os próximos 7 dias, auxiliando as equipes de logística e estoque de uma empresa varejista a tomar decisões mais informadas com base em dados históricos de vendas.

---
## **Estrutura do Projeto**
A estrutura do projeto é organizada da seguinte forma:

![](reports\estrutura_projeto.PNG)

### **1. Descrição dos Dados**
O arquivo utilizado é o vendas.csv, contendo os seguintes campos:

-   `sku`: Identificador único do produto
-   `data`: Data da venda no formato AAAA-MM-DD
-   `vendas`: Quantidade vendida no dia

Após o pré-processamento, também consideramos:

-   `unique_id`: Identificador alternativo para sku, usado pelo MLForecast
-   `ds`: Equivalente à data, usado pelo MLForecast
-   `y`: Equivalente a vendas, usado pelo MLForecast

Análise Exploratória e Considerações
-   O dataset contém vendas diárias durante um período de **3 meses**.

-   Cerca de **50% dos produtos têm menos de 12 observações**, o que indica séries temporais curtas e esparsas.

-   A análise de autocorrelação parcial (PACF) indicou significância em diversos lags (ex: 3, 4, 5, 7, 8, 9...), sugerindo dependência temporal útil.

-   Com isso, a escolha do modelo e a engenharia de features precisaram considerar produtos com séries curtas e longas separadamente, mas ao final, a abordagem **global com MLForecast** apresentou desempenho superior.

![](reports\distribuicao.png)

### **2. Fluxo de Trabalho**
O fluxo de trabalho do projeto segue as seguintes etapas:

1.  **Carregamento dos Dados**

    ```bash
        df = pd.read_csv("vendas.csv", parse_dates=["data"])
    ```

2.  **Pré-processamento**
    -   Renomeação de colunas para compatibilidade com o MLForecast
    -   Criação de features temporais como:
        -   Dia da semana
        -   Final de semana
        -   Dia do mês
        -   Mês do ano

3.  **Divisão Treino/Teste**
    -   Os últimos 7 dias foram usados como conjunto de teste.
    -   O restante foi usado para treino.

4.  **Treinamento de Modelos**
    Modelos utilizados no `MLForecast`
    -   `LGBMRegressor`
    -   `HistGradientBoostingRegressor`
    -   `RandomForestRegressor`

    Todos os modelos foram treinados usando:
    -   Features temporais
    -   Lags: 1, 3, 7, 14 dias
    -   Médias móveis e expandida

5.  **Interpretação do Modelo**
    -   Avaliação com métricas:
        -   MAE
        -   MAPE
        -   Erro RMSE ponderado pela quantidade vendida
    -   Escolha do melhor modelo com menor MAPE

6.  **Avaliação no Teste**
    Métrica usada:
    -   MAPE (Erro Percentual Absoluto Médio)
    -   Erro ponderado por quantidade de registros por produto


### ✅ **3.  Decisão Final**
A estratégia mais eficiente foi aplicar o **MLForecast globalmente**, treinando um modelo comum para todos os produtos. Isso foi justificado por:

-   Melhor desempenho nas métricas de erro
-   Facilidade na manutenção do pipeline
-   Melhora na capacidade de generalização, especialmente para produtos com poucos dados

Entre os modelos avaliados, o `LGBMRegressor` obteve o melhor desempenho com esses dados.

### **4.    Sugestões de melhorias**
-   Usar métricas ponderadas por cluster de produto (baixo/medio/alto volume)
-   Usar validação temporal com múltiplas janelas (ex: TimeSeriesSplit, ou backtesting com rolling window)
-   Salvar as métricas em reports/metrics.csv pode facilitar a rastreabilidade e auditoria.

## 🎁 **Bônus Implementados**
### 1. Visualização da Demanda Real x Prevista
Foi implementada uma função que permite ao usuário inserir o ID de um produto para visualizar a previsão dos próximos 7 dias.

![](reports\real_vs_previsto_conjunto.png)

### 2. Previsão por ID de Produto
Uma função permite que o usuário insira o ID do produto desejado para ver a previsão dos próximos 7 dias:

![](reports\real_vs_previsto_393.png)

## **Como Executar o Projeto**

### Pré-requisitos
- Python 3.8 ou superior.
- pip >= 22
- Bibliotecas listadas em `requirements.txt`

### Instale as dependências
```bash
   pip install -r requirements.txt
```

### Executar pipeline principal
```bash
   python main.py
```
### Fazer previsão com novos dados
**Observação:** Coloque os novos dados `.csv` na pasta `data/new/` com as colunas `unique_id`, `ds` e `y` (opcional).

Depois execute:
```bash
   python src/new_predict.py
```