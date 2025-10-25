import os
import logging
from src.feature_engineering import main as feature_main
from src.model_training import main as train_main
from src.new_predict import main as predict_main
import sys
from pathlib import Path

# Adiciona o diretório raiz (pai de src) ao sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# === CONFIGURAÇÃO DO LOGGING ===
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Iniciando pipeline principal...")

    # Etapa 1: Engenharia de Features
    try:
        logger.info("🔹 Etapa 1: Engenharia de Features")
        feature_main()
        logger.info("✅ Engenharia de features concluída com sucesso.")
    except Exception as e:
        logger.exception("❌ Erro na etapa de engenharia de features.")
        return

    # Etapa 2: Treinamento do Modelo
    try:
        logger.info("🔹 Etapa 2: Treinamento de Modelos")
        train_main()
        logger.info("✅ Treinamento de modelos concluído com sucesso.")
    except Exception as e:
        logger.exception("❌ Erro na etapa de treinamento de modelos.")
        return

    # Etapa 3: Predição com novos dados
    try:
        logger.info("🔹 Etapa 3: Predição em novos dados")
        predict_main()
        logger.info("✅ Predição concluída com sucesso.")
    except Exception as e:
        logger.exception("❌ Erro na etapa de predição.")
        return

    logger.info("🎉 Pipeline finalizado com sucesso.")

if __name__ == "__main__":
    main()


    
    