# Pipeline completo com nomes de colunas explícitos detectados no dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 0) Ler dados
# ------------------------------
path = "vandas_CodProduto_variaveis_socioecomicas.parquet"
df = pd.read_parquet(path)
df.reset_index(inplace=True)

# Nomes explícitos
date_col = "periodo"
code_col = "codigo"
qty_col  = "venda"

# Preparação
df[date_col] = pd.to_datetime(df[date_col].astype("str"), format="%Y-%m")
df.sort_values(date_col, inplace=True)

# Variáveis socioeconômicas (numéricas exceto venda e codigo)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
macro_cols = [c for c in num_cols if c not in [qty_col, code_col]]

# Agregar por mês por produto
df["ano_mes"] = df[date_col].values.astype("datetime64[M]")
grouped = df.groupby([code_col, "ano_mes"], as_index=False)[qty_col].sum()

# Matriz tempo x produto
mat = grouped.pivot(index="ano_mes", columns=code_col, values=qty_col).sort_index().fillna(0)

# ------------------------------
# 1) Clusterização por padrão temporal
# ------------------------------
def seasonal_strength(y):
    if (y>0).sum() < 6:
        return 0.0
    try:
        stl = STL(y, period=12, robust=True).fit()
        resid_var = np.var(stl.resid)
        seas_var  = np.var(stl.seasonal)
        if (seas_var + resid_var) == 0:
            return 0.0
        return max(0.0, min(1.0, 1 - resid_var/(seas_var + resid_var)))
    except Exception:
        return 0.0

features = []
for prod in mat.columns:
    y = mat[prod].astype(float).values
    mean = np.mean(y)
    std  = np.std(y)
    try:
        acf_vals = acf(y, nlags=min(24, len(y)-2), fft=True)
        acf12 = acf_vals[12] if len(acf_vals) > 12 else 0.0
    except Exception:
        acf12 = 0.0
    seas = seasonal_strength(mat[prod].astype(float))
    features.append([prod, mean, std, acf12, seas])

feat_df = pd.DataFrame(features, columns=["produto","mean","std","acf12","season_strength"]).set_index("produto")

scaler = StandardScaler()
X = scaler.fit_transform(feat_df.values)
k = 3 if mat.shape[1] >= 6 else min(2, mat.shape[1])
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X)
feat_df["cluster"] = labels

# ------------------------------
# 2) Macro mensal (média no mês)
# ------------------------------
macro_m = df.groupby("ano_mes", as_index=False)[macro_cols].mean().set_index("ano_mes").sort_index().interpolate()

# ------------------------------
# 3) Causalidade por cluster (série do grupo vs macro)
# ------------------------------
def granger_best_lag(y, x_df, maxlag=6):
    out = []
    for c in x_df.columns:
        sub = pd.concat([y, x_df[[c]]], axis=1).dropna()
        if len(sub) <= maxlag + 5:
            continue
        try:
            res = grangercausalitytests(sub, maxlag=maxlag, verbose=False)
            pvals = [res[L][0]["ssr_chi2test"][1] for L in range(1, maxlag+1)]
            lag = int(np.argmin(pvals)) + 1
            pmin = float(np.min(pvals))
            out.append((c, lag, pmin))
        except Exception:
            continue
    return pd.DataFrame(out, columns=["variavel","lag_ideal","p_min"]).sort_values("p_min")

results_by_cluster = {}
forecasts_by_cluster = {}
h = 3  # horizonte (meses)

for cl in sorted(feat_df["cluster"].unique()):
    prods = feat_df.index[feat_df["cluster"]==cl].tolist()
    grp_series = mat[prods].sum(axis=1)
    grp_series.name = "vendas_grupo"
    df_grp = pd.concat([grp_series, macro_m], axis=1).dropna()
    y = df_grp[["vendas_grupo"]]
    X = df_grp.drop(columns=["vendas_grupo"])

    caus = granger_best_lag(y, X, maxlag=6)
    caus_sig = caus[caus["p_min"] < 0.05].head(6)
    results_by_cluster[cl] = caus_sig

    # Construir exógenas defasadas
    exog = pd.DataFrame(index=df_grp.index)
    for _, row in caus_sig.iterrows():
        var = row["variavel"]; lag = int(row["lag_ideal"])
        exog[f"{var}_lag{lag}"] = X[var].shift(lag)

    data = pd.concat([y, exog], axis=1).dropna()
    y_fit = data["vendas_grupo"]
    X_fit = data.drop(columns=["vendas_grupo"])

    # SARIMAX
    if len(y_fit) < 18 or X_fit.shape[1]==0:
        model = SARIMAX(y_fit, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        res_model = model.fit(disp=False)
        fut_exog = None
    else:
        model = SARIMAX(y_fit, exog=X_fit, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        res_model = model.fit(disp=False)
        # exógenas futuras por média dos últimos 3 meses
        last_idx = X_fit.index[-1]
        future_idx = pd.date_range(start=last_idx, periods=h+1, freq="MS")[1:]
        fut_exog = pd.DataFrame(index=future_idx)
        for col in X_fit.columns:
            fut_exog[col] = X_fit[col].iloc[-3:].mean()

    if fut_exog is not None:
        fc = res_model.get_forecast(steps=h, exog=fut_exog)
    else:
        fc = res_model.get_forecast(steps=h)

    forecasts_by_cluster[cl] = fc.predicted_mean.rename(f"cluster_{cl}")

# ------------------------------
# 4) Consolidar previsões e distribuir por produto via share
# ------------------------------
if forecasts_by_cluster:
    fc_df = pd.concat(forecasts_by_cluster.values(), axis=1)
    fc_df["total_previsto"] = fc_df.sum(axis=1)
else:
    fc_df = pd.DataFrame()

# Shares dos últimos 6 meses por cluster
window = 6
shares = {}
for cl in sorted(feat_df["cluster"].unique()):
    prods = feat_df.index[feat_df["cluster"]==cl].tolist()
    sub = mat[prods].iloc[-window:]
    s = (sub.mean() / sub.mean().sum()).fillna(0.0)
    shares[cl] = s

product_fc = {}
for cl, fc_series in forecasts_by_cluster.items():
    s = shares.get(cl, None)
    if s is None or s.sum()==0:
        continue
    for prod, w in s.items():
        product_fc.setdefault(prod, pd.Series(0, index=fc_series.index))
        product_fc[prod] = product_fc[prod].add(fc_series * float(w), fill_value=0.0)

prod_fc_df = pd.DataFrame(product_fc) if product_fc else pd.DataFrame()

# ------------------------------
# 5) Salvar saídas
# ------------------------------
clusters_path = "resultados/00_clusters_produtos.csv"
feat_df.reset_index().to_csv(clusters_path, index=False)

sig_path = "resultados/01_variaveis_significativas_por_cluster.csv"
rows = []
for cl, dfc in results_by_cluster.items():
    for _, r in dfc.iterrows():
        rows.append({"cluster": cl, "variavel": r["variavel"], "lag_ideal": int(r["lag_ideal"]), "p_min": float(r["p_min"])})
pd.DataFrame(rows).to_csv(sig_path, index=False)

fc_path = "resultados/02_previsoes_por_cluster.csv"
if len(fc_df)>0:
    fc_df.to_csv(fc_path)

prod_fc_path = "resultados/03_previsoes_por_produto.csv"
if len(prod_fc_df)>0:
    prod_fc_df.to_csv(prod_fc_path)

# ------------------------------
# 6) Gráficos simples (1 por figura, sem estilos)
# ------------------------------
plt.figure()
feat_df["cluster"].value_counts().sort_index().plot(kind="bar")
plt.title("Quantidade de produtos por cluster")
plt.xlabel("Cluster")
plt.ylabel("Nº de produtos")
plt.tight_layout()
plt.savefig("resultados/grafico_qtd_produtos_por_cluster.png", dpi=140)
plt.close()

if forecasts_by_cluster:
    plt.figure()
    fc_df["total_previsto"].plot(marker="o")
    plt.title("Previsão total (soma dos clusters)")
    plt.xlabel("Período futuro")
    plt.ylabel("Quantidade prevista")
    plt.tight_layout()
    plt.savefig("resultados/grafico_previsao_total.png", dpi=140)
    plt.close()

outputs = {
 "clusters_csv": clusters_path,
 "significativas_csv": sig_path,
 "previsoes_cluster_csv": fc_path,
 "previsoes_produto_csv": prod_fc_path,
 "grafico_clusters": "resultados/grafico_qtd_produtos_por_cluster.png",
 "grafico_previsao_total": "resultados/grafico_previsao_total.png",
 "exog_usadas_por_cluster": {int(k): v["variavel"].tolist() if len(v)>0 else [] for k,v in results_by_cluster.items()}
}
outputs
