import pandas as pd
import logging
from pathlib import Path
from src.utils.utils import localizar_modelo_salvo, localizar_arquivo_dados, load_model
from src.feature_engineering import create_features

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/new_predict.log"),
        logging.StreamHandler()
    ]
)

N_DIAS_PREVISAO = 7
CAMINHO_SAIDA = "resultados/predicoes_novas.csv"

def main():
    logging.info("📦 Carregando modelo...")
    try:
        CAMINHO_MODELO = localizar_modelo_salvo()
        model = load_model(CAMINHO_MODELO)
    except FileNotFoundError as e:
        logging.error(f"❌ {e}")
        logging.warning("⚠️ Encerrando etapa de predição por falta de modelo.")
        return

    logging.info("📥 Verificando dados novos para previsão...")
    try:
        CAMINHO_DADOS = localizar_arquivo_dados()
        df = pd.read_csv(CAMINHO_DADOS, parse_dates=["ds"])
    except FileNotFoundError as e:
        logging.warning(f"⚠️ {e}")
        logging.info("⚠️ Nenhum dado novo encontrado. Etapa de previsão será ignorada.")
        return
    
if __name__ == "__main__":
    main()