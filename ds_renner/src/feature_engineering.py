# Bibliotecas
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#Constantes
# Constantes
PROCESSED_DATA_PATH = "data/raw/vendas.csv"  # Caminho para os dados processados
ENGINEERED_DATA_PATH = "data/processed/vendas_engineered.parquet"  # Caminho para salvar os dados com novas features

# Função para carregar os dados processados
def load_processed_data(data_path):
    """
    Carrega os dados processados a partir de um arquivo CSV.
    """
    logger.info(f"Carregando dados processados de {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["data_venda"])
    data = data.rename(columns={"data_venda": "ds", "sku": "unique_id", "venda": "y"})
    return data

# 
def create_features(data):
    """
    Cria novas features a partir dos dados processados.
    """
    logger.info("Criando novas features...")

    # Criar features agregadas para cada produto
    product_features = data.groupby('unique_id').agg({
        'y': ['mean', 'std', 'count'],
    }).fillna(0)

    # Ajustar os nomes das colunas (remover multi-index)
    product_features.columns = ['y_mean', 'y_std', 'y_count']

    # Padronização dos dados
    scaler = StandardScaler()
    product_features_scaled = scaler.fit_transform(product_features)

    # Clusterização final
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(product_features_scaled)

    # Mapear os clusters de volta para o DataFrame original
    data['product_cluster'] = data['unique_id'].map(
        dict(zip(product_features.index, clusters)))
    
    # Atribuir pesos inversamente proporcionais à frequência
    product_weights = data['unique_id'].value_counts(normalize=True).rdiv(1)
    data['sample_weight'] = data['unique_id'].map(product_weights)
    
    return data

# Função para salvar os dados com novas features
def save_engineered_data(data, save_path):
    """
    Salva os dados com novas features em um arquivo parquet.
    """
    logger.info(f"Salvando dados com novas features em {save_path}...")
    data.to_parquet(save_path, index=False)

# Função principal
def main():
    """
    Função principal para executar o pipeline de engenharia de features.
    """
    try:
        # 1. Carregar dados processados
        processed_data = load_processed_data(PROCESSED_DATA_PATH)

        # 2. Criar novas features
        engineered_data = create_features(processed_data)
        
        # 3. Salvar dados com novas features
        save_engineered_data(engineered_data, ENGINEERED_DATA_PATH)

        logger.info("Engenharia de features concluída com sucesso!")

    except Exception as e:
        logger.error(f"Erro durante a engenharia de features: {e}")

# Executar o pipeline
if __name__ == "__main__":
    main()