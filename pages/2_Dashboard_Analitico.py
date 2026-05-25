# =============================================================================
# pages/2_Dashboard_Analitico.py
# -----------------------------------------------------------------------------
# Dashboard analitico exigido pelo Tech Challenge. Apresenta os principais
# insights da base de obesidade em uma visao de negocio voltada para a
# equipe medica. Todas as visualizacoes usam Plotly para serem interativas.
# =============================================================================

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Permite importar funcoes do diretorio raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_model import (  # noqa: E402
    METRICS_PATH,
    TARGET_CLASSES,
    add_engineered_features,
    clean_decimal_categoricals,
    load_raw_data,
)

# -----------------------------------------------------------------------------
# Configuracao da pagina
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analitico - Obesidade",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard Analitico - Obesidade")
st.caption(
    "Visao analitica para apoiar a tomada de decisao da equipe medica. "
    "Todos os graficos sao interativos: passe o mouse para detalhes e use "
    "a legenda para filtrar series."
)


# -----------------------------------------------------------------------------
# Carregamento e preparacao da base
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Carrega o CSV bruto, aplica limpeza basica e cria as features
    derivadas. Mantemos a base sem encodings para que os graficos exibam
    rotulos legiveis."""
    df = load_raw_data()
    df = clean_decimal_categoricals(df)
    df = add_engineered_features(df)
    # Adiciona uma faixa de IMC (categorica) so para algumas visualizacoes.
    df["IMC_Faixa"] = pd.cut(
        df["imc"],
        bins=[0, 18.5, 25, 30, 35, 40, 100],
        labels=[
            "Abaixo (<18.5)",
            "Normal (18.5-25)",
            "Sobrepeso (25-30)",
            "Obesidade I (30-35)",
            "Obesidade II (35-40)",
            "Obesidade III (>=40)",
        ],
    )
    # Faixa etaria para apoiar analises por geracao.
    df["Faixa_Etaria"] = pd.cut(
        df["Age"],
        bins=[0, 25, 40, 100],
        labels=["Jovem (<=25)", "Adulto (26-40)", "Maduro (>40)"],
    )
    return df


@st.cache_data
def load_metrics_report():
    """Carrega o relatorio de metricas gerado por train_model.py se ele
    existir. Usado na secao de qualidade do modelo."""
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


df = load_dataset()
metrics = load_metrics_report()

# -----------------------------------------------------------------------------
# Variaveis Globais e Dicionarios de Traducao
# -----------------------------------------------------------------------------
# Define ordem fixa para as classes nos graficos (do mais leve ao mais grave).
ORDERED_CLASSES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

# Dicionários de tradução comuns para todo o painel
obesity_labels = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade Grau I",
    "Obesity_Type_II": "Obesidade Grau II",
    "Obesity_Type_III": "Obesidade Grau III"
}
ordered_labels_pt = [obesity_labels[c] for c in ORDERED_CLASSES]


# -----------------------------------------------------------------------------
# Filtros laterais
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filtros")
    selected_genders = st.multiselect(
        "Genero",
        options=sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique()),
    )
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider(
        "Faixa de idade",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )
    selected_history = st.multiselect(
        "Historico familiar de sobrepeso",
        options=sorted(df["family_history"].unique()),
        default=sorted(
