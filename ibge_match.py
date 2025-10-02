import pandas as pd
import requests
import time
import urllib.parse
from datetime import datetime

# dados originais 
df = pd.read_csv("df_vendas.csv", sep=";")
#df.xmun = df.xmun.str.replace("TOLEDO", "TOLEDO (PR)")
lista_municipio = pd.read_excel("lista_municipio.xlsx")
df = df.merge(lista_municipio[['xmun', 'nome_correto']], on='xmun', how='left')
# normalizar os nomes dos municipios e adicionando codigo
df[["cod_ibge", "nome_oficial", "uf_oficial"]] = df["nome_correto"].apply(lambda x: pd.Series(get_cod_ibge(x)))
df["cod_ibge"] = df.cod_ibge.astype("int64")
df["cod_ibge"] = df.cod_ibge.astype("str")

# criando a coluna data
df["data"] = pd.to_datetime(
    dict(year=df["ano"], month=df["mes"], day=1),
    errors="coerce"
)
# em string no formato
df["data"] = df["data"].dt.strftime("%d/%m/%Y")


# dados credito agricola
# Lista de estados desejados
municipios = df.cod_ibge.unique()

SELECT = (
    "Municipio,nomeProduto,MesEmissao,AnoEmissao,cdPrograma,cdSubPrograma,"
    "cdFonteRecurso,cdTipoSeguro,cdEstado,VlCusteio,cdProduto,codCadMu,"
    "Atividade,cdModalidade,codIbge,AreaCusteio"
)
BASE = "https://olinda.bcb.gov.br/olinda/servico/SICOR/versao/v2/odata/CusteioMunicipioProduto"

def consulta_municipio(cod_ibge: str,
                       connect_timeout=10,
                       read_timeout=120,
                       sleep_between_pages=0.2) -> pd.DataFrame:
    # filtro com codIbge ENTRE aspas simples e $filter primeiro
    url = (
        f"{BASE}?$filter=codIbge%20eq%20'{cod_ibge}'"
        f"&$format=json&$select={SELECT}&$top=5000"
    )
    dfs = []
    while url:
        resp = requests.get(url, timeout=(connect_timeout, read_timeout))
        resp.raise_for_status()
        payload = resp.json()
        dfs.append(pd.DataFrame(payload.get("value", [])))
        url = payload.get("@odata.nextLink") or payload.get("odata.nextLink")
        if url:
            time.sleep(sleep_between_pages)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- seu fluxo (mantendo os códigos IBGE como string!) ---
# municipios = df.cod_ibge.astype(str).unique()   # <- evite int pra não perder zeros à esquerda
# Se já tem: municipios = df['cod_ibge'].astype(str).unique()

def consultar_lista(municipios_cod_ibge):
    resultados = []
    for m in map(str, municipios_cod_ibge):
        try:
            df_m = consulta_municipio(m)
            if not df_m.empty:
                resultados.append(df_m)
            else:
                print(f"[INFO] Sem dados para {m}")
        except requests.RequestException as e:
            print(f"[WARN] Falha ao consultar {m}: {e}")
        time.sleep(0.4)  # pausa curta entre municípios
    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()

todos = consultar_lista(municipios)

#adicinar uma coluna data
todos["data"] = pd.to_datetime(
    dict(year=todos["AnoEmissao"], month=todos["MesEmissao"], day=1),
    errors="coerce"
)
# em string no formato
todos["data"] = todos["data"].dt.strftime("%d/%m/%Y")

# agrupar os meses
df_custeio = todos[["data", "codIbge", "VlCusteio", "AreaCusteio"]].groupby(["data", "codIbge"]).agg(
    soma_custeio = ("VlCusteio", "sum"),
    soma_area_custeio = ("AreaCusteio", "sum")
).reset_index()

df_custeio.columns = ["data", "cod_ibge", "VlCusteio", "AreCusteio"]

# merge com os dados originais
df = df.merge(df_custeio, on=['cod_ibge', 'data'], how='left')

#importar produção agricola
df_prod_agricola = pd.read_excel('producao_agricola.xlsx', 
                                 index_col=[0,1],
                                 engine='openpyxl',
                                 dtype={"codigo": "string"},
                                 converters={4: lambda x: None if pd.isna(x) else int(x)})
# renomear a coluna
df_prod_agricola.rename(columns={'codigo': 'cod_ibge'}, inplace=True)

df_prod_agricola = df_prod_agricola[df_prod_agricola.cod_ibge.isin(municipios) & df_prod_agricola.quant_produzida_t.notna()]

lista = [""]
for i in df_prod_agricola.index:
    if lista[-1] != i[1]:
        lista.append(i[1])

df_prod_agricola.reset_index().to_csv("df_prod_agricola.csv", sep=";")

## importancia das culturas
import_cultura = df_prod_agricola.groupby(['municipio', 'produto']).agg(
    per_utilizado = ('percentual_utilizado', "mean")    
).reset_index().sort_values(by=['municipio', 'per_utilizado'], ascending=False)

df_culturas = pd.DataFrame(columns=list(import_cultura.columns))
for i in import_cultura.municipio.unique():
    df_rank = import_cultura[import_cultura.municipio == i].reset_index(drop=True)
    for i in df_rank.index:
        if df_rank[:i].per_utilizado.sum() > 90:
            df_culturas = pd.concat([df_culturas, df_rank[:i]], axis=0)
            break

df_culturas.produto.unique()

# preco do feijao
df_feijao = pd.read_csv("preco\\preco_feijao.csv", sep=";") #, parse_dates=['data']
df_feijao.rename(columns={"uf": "uf_oficial"}, inplace=True)
#merge com os precos de feijão
df = df.merge(df_feijao, on=['data', 'uf_oficial'], how='left')

# preço da laranja
df_laranja = pd.read_csv("preco\\preco_laranja.csv", sep=";")
df_laranja.rename(columns={"uf": "uf_oficial"}, inplace=True)
# merge com os precos de laranja
df = df.merge(df_laranja, on=['data', 'uf_oficial'], how='left')

# preco do trigo
df_trigo = pd.read_excel("preco\\preco_trigo.xlsx")
df_trigo['data'] = pd.to_datetime(df_trigo['data'], format='%d/%m/%Y')
df_trigo['mes'] = [i.month for i in df_trigo.data]
df_trigo['ano'] = [i.year for i in df_trigo.data]
df_trigo = df_trigo.groupby(['uf', 'mes', 'ano']).agg(
    preco_trigo = ('preco_trigo', 'mean')
).reset_index().rename(columns={"uf": "uf_oficial"})
# criando a coluna data
df_trigo["data"] = pd.to_datetime(
    dict(year=df_trigo["ano"], month=df_trigo["mes"], day=1),
    errors="coerce"
)
# em string no formato
df_trigo["data"] = df_trigo["data"].dt.strftime("%d/%m/%Y")
# merge com os precos de laranja
df = df.merge(df_trigo[['data', 'uf_oficial', 'preco_trigo']], on=['data', 'uf_oficial'], how='left')

# preço de soja
df_soja = pd.read_parquet("d:\\documentos\\capacidade_pagamento\\data\\processed\\historico_preco_soja.parquet")
df_soja["data"] = df_soja["data"].dt.strftime("%d/%m/%Y")
df_soja.rename(columns={"uf": "uf_oficial", "preco_medio": "preco_soja"}, inplace=True)
#merge com os precos do soja
df = df.merge(df_soja[["data", "uf_oficial", "preco_soja"]], on=["data", "uf_oficial"], how="left")

# preço do milho
df_milho = pd.read_parquet("d:\\documentos\\capacidade_pagamento\\data\\processed\\historico_preco_milho.parquet")
df_milho["data"] = df_milho["data"].dt.strftime("%d/%m/%Y")
df_milho.rename(columns={"uf": "uf_oficial", "preco_medio": "preco_milho"}, inplace=True)
# merge com precos do milho
df = df.merge(df_milho[["data", "uf_oficial", "preco_milho"]], on=["data", "uf_oficial"], how="left")

df.to_parquet("df_vendas.parquet", index=False)
