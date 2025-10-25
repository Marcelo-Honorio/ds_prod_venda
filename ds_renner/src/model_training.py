
# biblitecas
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import os
from src.utils.utils import erro_ponderado_por_qtd
from src.feature_engineering import create_features
from mlforecast import MLForecast
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns 


# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#Caminhos e valores
DATA_PATH = "data/processed/vendas_engineered.parquet"  # Caminho para os dados processados
MODEL_PATH = "models/churn_model.pkl"            # Caminho para salvar o modelo treinado
RANDOM_STATE = 42                                # Semente para reprodutibilidade

# Função para carregar os dados
def load_data_parquet(data_path):
    """
    Carrega os dados processados a partir de um arquivo CSV.
    """
    logger.info(f"Carregando dados de {data_path}...")
    data = pd.read_parquet(data_path)
    return data

# Funções para date_features
def is_weekend(dates):
    """É final de semana"""
    return dates.dayofweek.isin([5, 6]).astype(int)

def day_month(dates):
    """dia do mês"""
    return dates.day

def month(dates):
    """mes do ano"""
    return dates.month

# Função para dividir os dados em treino e teste
def split_data(data, dias_teste):
    """
    Divide os dados em conjuntos de treino e teste.
    """
    logger.info("Dividindo dados em treino e teste...")
    train = (
        data.groupby("unique_id", group_keys=False)
        .apply(lambda x: x.iloc[:-dias_teste])
        .reset_index(drop=True)
    )

    test = (
        data.groupby("unique_id", group_keys=False)
        .apply(lambda x: x.iloc[-dias_teste:])
        .reset_index(drop=True)
    )
    return train, test

# Treinar modelo com MLForecast
def rodar_mlforecast(df_train):
    """
    Treina o modelo com MLForecast.
    """
    logger.info("Treinando com MLForecast...")
    modelos = [
        ('LGBM', lgb.LGBMRegressor(random_state=42, verbosity=-1, min_child_samples=30, reg_alpha=0.2, reg_lambda=0.2, n_estimators=100)),
        ('HGBR', HistGradientBoostingRegressor(random_state=42, min_samples_leaf=30, l2_regularization=0.2, max_iter=100)),
        ('RF', RandomForestRegressor(random_state=42, n_estimators=100))
    ]

    fcst = MLForecast(
        models=[m[1] for m in modelos],
        freq='D',
        lags=[1, 3, 7],
        lag_transforms={
            1: [ExpandingMean()],
            3: [RollingMean(window_size=3)],
            7: [RollingMean(window_size=7)],
            14: [RollingMean(window_size=14)]
        },
        date_features=["dayofweek", "dayofyear", is_weekend, day_month, month]
    )

    fcst.fit(
        df_train,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        weight_col="sample_weight",
        static_features=["product_cluster"]
    )

    return fcst, modelos


# Função para avaliar o modelo
def evaluate_model(fcst, df_test, modelos):
    """
    Avalia o modelo usando métricas como MAE, MAPE, Erro Ponderado.
    """
    logger.info("Avaliando o modelo...")
    # Fazer previsões
    pred = fcst.predict(7)
    df_aval = df_test[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="inner")

    resultados = []
    best_metrics = {
        'model_name': None,
        'model': None,
        'mae': float('inf'),
        'mape': float('inf'),
        'erro_ponderado': float('inf')
    }

    model_map = {m[0]: m[1] for m in modelos}

    for col in pred.columns:
        for prefix in model_map:
            if col.startswith(prefix):
                mae = mean_absolute_error(df_aval["y"], df_aval[col])
                mape = mean_absolute_percentage_error(df_aval["y"], df_aval[col])
                erro_pond = erro_ponderado_por_qtd(df_aval, col)

                resultados.append({
                    'model_name': col,
                    'mae': mae,
                    'mape': mape,
                    'erro_ponderado': erro_pond
                })

                print(f"{col}:\n MAE: {mae:.2f}\n MAPE: {mape:.2%}\n Erro Ponderado: {erro_pond:.2f}\n")

                if mape < best_metrics['mape']:
                    best_metrics.update({
                        'model_name': col,
                        'model': model_map[prefix],
                        'mae': mae,
                        'mape': mape,
                        'erro_ponderado': erro_pond
                    })

    return pred, resultados, best_metrics

# Salvar o melhor modelo
def salvar_melhor_modelo(best_metrics, save_path='models'):
    if best_metrics['model_name'] is None:
        print("Nenhum modelo foi selecionado como o melhor.")
        return

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Remover modelos anteriores
    for arquivo in os.listdir(save_path):
        if arquivo.endswith('_best_model.pkl'):
            os.remove(os.path.join(save_path, arquivo))
            print(f"🧹 Modelo antigo removido: {arquivo}")

    # Preparar informações do novo modelo
    model_info = {
        'model': best_metrics['model'],
        'model_name': best_metrics['model_name'],
        'metrics': {
            'mae': best_metrics['mae'],
            'mape': best_metrics['mape'],
            'erro_ponderado': best_metrics['erro_ponderado']
        },
        'mlforecast_config': {
            'freq': 'D',
            'lags': [1, 3, 7],
            'lag_transforms': {
                1: [ExpandingMean()],
                3: [RollingMean(window_size=3)],
                7: [RollingMean(window_size=7)],
                14: [RollingMean(window_size=14)]
            },
            'date_features': ["dayofweek", "dayofyear", is_weekend, day_month, month]
        },
        'fit_params': {
            'id_col': "unique_id",
            'time_col': "ds", 
            'target_col': "y",
            'static_features': ["product_cluster"]
        }
    }

    # Caminho do novo modelo
    model_path = f"{save_path}/{best_metrics['model_name']}_best_model.pkl"
    joblib.dump(model_info, model_path)

    print(f"\nMelhor modelo salvo em: {model_path}")
    print(f"Modelo: {best_metrics['model_name']} com MAPE: {best_metrics['mape']:.2%}")


# Previsão com o melhor modelo
def forecast_with_model(model_dict: dict, df: pd.DataFrame, n_passos: int):
    fcst = MLForecast(
        models=[model_dict["model"]],
        freq='D',
        lags=[1, 3, 7],
        lag_transforms={
            1: [ExpandingMean()],
            3: [RollingMean(window_size=3)],
            7: [RollingMean(window_size=7)]
        },
        date_features=["dayofweek", "dayofyear", is_weekend, day_month, month]
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        weight_col="sample_weight",
        static_features=["product_cluster"]
    )
    pred = fcst.predict(n_passos)
    return pred

# =============================================
# BÔNUS 1: Visualização demanda real vs prevista
# =============================================
def plot_real_vs_previsto_conjunto(df_real: pd.DataFrame, df_pred: pd.DataFrame, model_col: str):
    """
    Plota comparação agregada entre valores reais e previstos para todo o conjunto de teste
    
    Args:
        df_real: DataFrame com valores reais (deve conter 'ds', 'y')
        df_pred: DataFrame com previsões (deve conter 'ds' e coluna do modelo)
        model_col: Nome da coluna com as previsões
    """
    # Merge e agregação por data
    df_plot = df_real.merge(df_pred, on="ds", how="inner")
    df_agg = df_plot.groupby("ds").agg({'y':'sum', model_col:'sum'}).reset_index()
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df_agg["ds"], df_agg["y"], label="Real", marker="o", linewidth=2)
    plt.plot(df_agg["ds"], df_agg[model_col], label="Previsto", marker="x", linestyle="--", linewidth=2)
    
    # Formatação
    plt.title("Demanda Real vs Prevista - Conjunto Completo de Teste", pad=20)
    plt.xlabel("Data", labelpad=10)
    plt.ylabel("Demanda Total", labelpad=10)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar e mostrar
    Path("reports/figures").mkdir(exist_ok=True)
    plt.savefig("reports/figures/real_vs_previsto_conjunto.png", dpi=300)
    plt.show()

# =============================================
# BÔNUS 2: Previsão para produto específico
# =============================================
def prever_produto_especifico(model_dict: dict, df_completo: pd.DataFrame, product_id: str, dias_previsao: int = 7):
    """
    Gera previsão para um produto específico e plota resultados
    
    Args:
        model_dict: Modelo carregado (do joblib)
        df_completo: DataFrame com dados históricos completos
        product_id: ID do produto desejado
        dias_previsao: Horizonte de previsão
        
    Returns:
        DataFrame com previsões e mostra gráfico
    """
    # Filtrar dados do produto
    df_produto = df_completo[df_completo["unique_id"] == product_id].copy()
    
    if df_produto.empty:
        raise ValueError(f"Produto {product_id} não encontrado!")
    
    # Configurar modelo
    fcst = MLForecast(
        models=[model_dict["model"]],
        freq='D',
        lags=model_dict["mlforecast_config"]["lags"],
        lag_transforms=model_dict["mlforecast_config"]["lag_transforms"],
        date_features=model_dict["mlforecast_config"]["date_features"]
    )
    
    # Treinar e prever
    fcst.fit(df_produto, id_col="unique_id", time_col="ds", target_col="y")
    pred = fcst.predict(dias_previsao)
    
    # Plotar histórico + previsão
    plt.figure(figsize=(12, 6))
    
    # Últimos 30 dias do histórico
    historico = df_produto.tail(30)
    plt.plot(historico["ds"], historico["y"], label="Histórico", marker="o", color='blue')
    
    # Previsão
    plt.plot(pred["ds"], pred[model_dict["model_name"]], label="Previsão", marker="x", color='red', linestyle="--")
    
    # Formatação
    plt.title(f"Previsão de Demanda - Produto {product_id}", pad=20)
    plt.xlabel("Data", labelpad=10)
    plt.ylabel("Demanda", labelpad=10)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar
    Path("reports/figures").mkdir(exist_ok=True)
    plt.savefig(f"reports/figures/previsao_{product_id}.png", dpi=300)
    plt.show()
    
    return pred

# Função principal
def main():
    """
    Função principal para executar o pipeline de treinamento.
    """
    try:
        # 1. Carregar os dados
        data = load_data_parquet(DATA_PATH)

        # 2. Dividir os dados em treino e teste
        train, test = split_data(data = data, dias_teste=7)

        # 3. Treinar o modelo
        fcst, modelos = rodar_mlforecast(train)

        # 4. Avaliar o modelo
        pred_test, _, best_metrics = evaluate_model(fcst, test, modelos)

        # 5. Salvar o modelo
        salvar_melhor_modelo(best_metrics, save_path='models')

        # Bônus 1 - Visualização do conjunto de teste
        plot_real_vs_previsto_conjunto(
            df_real=test,
            df_pred=pred_test,
            model_col=best_metrics["model_name"]  # Nome do melhor modelo
        )
    
        # Bônus 2 - Previsão para produto específico
        produto = input("\nDigite o ID do produto para previsão: ").strip()
        
        try:
            prever_produto_especifico(
                model_dict=joblib.load(f"models/{best_metrics['model_name']}_best_model.pkl"),
                df_completo=data,
                product_id=produto,
                dias_previsao=14
            )
        except Exception as e:
            logger.error(f"Erro ao prever produto: {e}")

        logger.info("Pipeline de treinamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento do modelo: {e}")

if __name__ == "__main__":
    main()