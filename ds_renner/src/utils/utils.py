import numpy as np
from sklearn.metrics import root_mean_squared_error
import os
from pathlib import Path
import joblib


# Treinar modelo com MLForecast e avaliação com erro médio ponderado
def erro_ponderado_por_qtd(df_aval, col_pred):
    erros = []
    for uid, grupo in df_aval.groupby("unique_id"):
        if len(grupo) < 1:
            continue
        erro = root_mean_squared_error(grupo["y"], grupo[col_pred])
        peso = len(grupo)
        erros.append((erro, peso))

    total_pesos = sum(p for _, p in erros)
    if total_pesos == 0:
        return np.nan

    erro_ponderado = sum(e * p for e, p in erros) / total_pesos
    return erro_ponderado

#localizar o modelo salvo
def localizar_modelo_salvo(pasta_modelos='models'):
    """Encontra automaticamente o arquivo *_best_model.pkl"""
    for nome_arquivo in os.listdir(pasta_modelos):
        if nome_arquivo.endswith('_best_model.pkl'):
            return str(Path(pasta_modelos) / nome_arquivo)
    raise FileNotFoundError("❌ Nenhum modelo *_best_model.pkl encontrado na pasta de modelos.")

# localizar novas predições
def localizar_arquivo_dados(pasta_dados='data/new'):
    """Encontra automaticamente o primeiro arquivo .csv em data/new/"""
    arquivos = [f for f in os.listdir(pasta_dados) if f.endswith('.csv')]
    if not arquivos:
        raise FileNotFoundError("❌ Nenhum arquivo .csv encontrado em data/new/")
    return str(Path(pasta_dados) / arquivos[0])

# função para carregar o modelo salvo
def load_model(path: str):
    return joblib.load(path)

