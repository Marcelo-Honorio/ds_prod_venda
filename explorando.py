import pandas as pd
import re, unicodedata, requests
import urllib.parse

try:
    from ftfy import fix_text
except Exception:
    def fix_text(s): return s  # fallback

# ---------------- Utils ----------------
UF_SIGLAS = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT',
             'PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))

def normalize_mun(nome: str) -> str:
    """Normaliza: corrige encoding, remove UF explícita, tira símbolos, upper sem acento."""
    if pd.isna(nome):
        return None
    s = fix_text(str(nome)).strip()
    # remove "- UF" ou "(UF)"
    s = re.sub(r'\s*-\s*[A-Z]{2}\b', '', s)
    s = re.sub(r'\([A-Z]{2}\)', '', s)
    # limpa símbolos estranhos
    s = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return strip_accents(s).upper()

def extract_uf_hint(raw: str):
    """Tenta extrair UF de sufixo '- PR' ou '(PR)' antes de normalizar."""
    if raw is None:
        return None
    m = re.search(r'\b-\s*([A-Z]{2})\b', str(raw))
    if not m:
        m = re.search(r'\(([A-Z]{2})\)', str(raw))
    if m and m.group(1) in UF_SIGLAS:
        return m.group(1)
    return None

# ---------------- IBGE robusto (por UF) ----------------
def carregar_tabela_ibge() -> pd.DataFrame:
    """
    Evita a cadeia microrregiao/mesorregiao/UF (que às vezes vem faltando).
    Baixa por UF, garante que sempre teremos a sigla da UF.
    """
    rows = []
    for uf in UF_SIGLAS:
        url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json() or []
        for it in data:
            nome = it.get('nome')
            cod = it.get('id')
            if not nome or not cod:
                continue
            key = normalize_mun(nome)  # já gera a chave normalizada
            rows.append((key, uf, cod, nome))
    return pd.DataFrame(rows, columns=["key","uf","cod_ibge","nome_oficial"])

ibge_df = carregar_tabela_ibge()

# Índices auxiliares para acelerar a busca
# exato: key -> lista de (uf, cod, nome)
by_key = {}
for _, row in ibge_df.iterrows():
    by_key.setdefault(row['key'], []).append((row['uf'], row['cod_ibge'], row['nome_oficial']))

# ---------------- Busca com re.search ----------------
def get_cod_ibge(municipio: str):
    """
    Normaliza e tenta:
      1) match exato (com UF da string se houver)
      2) match exato (sem UF)
      3) regex parcial (priorizando UF da string se houver)
    Retorna (cod_ibge, nome_oficial, uf) ou (None, None, None) se não achar.
    """
    if not municipio:
        return None, None, None

    uf_hint = extract_uf_hint(municipio)
    nome_norm = normalize_mun(municipio)

    # 1) Exato com UF
    if nome_norm in by_key and uf_hint:
        for uf, cod, nome in by_key[nome_norm]:
            if uf == uf_hint:
                return cod, nome, uf

    # 2) Exato sem UF (se houver homônimos, fica o primeiro)
    if nome_norm in by_key:
        uf, cod, nome = by_key[nome_norm][0]
        return cod, nome, uf

    # 3) Parcial por regex (prioriza UF hint)
    # monta padrão seguro para regex (escapa o nome)
    pattern = re.compile(re.escape(nome_norm))
    candidates = ibge_df

    if uf_hint:
        candidates = candidates[candidates['uf'] == uf_hint]

    # primeiro tenta na UF sugerida (se houver), senão no geral
    for _, row in candidates.iterrows():
        if pattern.search(row['key']):
            return row['cod_ibge'], row['nome_oficial'], row['uf']

    if uf_hint:
        # tenta geral se não achou na UF sugerida
        for _, row in ibge_df.iterrows():
            if pattern.search(row['key']):
                return row['cod_ibge'], row['nome_oficial'], row['uf']

    return None, None, None

