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
        default=sorted(df["family_history"].unique()),
    )
    selected_classes = st.multiselect(
        "Niveis de obesidade",
        options=ORDERED_CLASSES,
        default=ORDERED_CLASSES,
    )
    st.caption(
        "Os filtros se aplicam a todos os graficos abaixo. Limpe um filtro "
        "para voltar a base completa."
    )

# Aplica os filtros.
df_filt = df[
    df["Gender"].isin(selected_genders)
    & df["Age"].between(age_range[0], age_range[1])
    & df["family_history"].isin(selected_history)
    & df["Obesity"].isin(selected_classes)
].copy()

if df_filt.empty:
    st.warning("Nenhum registro corresponde aos filtros selecionados.")
    st.stop()


# -----------------------------------------------------------------------------
# KPIs principais
# -----------------------------------------------------------------------------
st.subheader("Indicadores principais")

total_pacientes = len(df_filt)
imc_medio = df_filt["imc"].mean()
pct_obesidade = (
    df_filt["Obesity"]
    .isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"])
    .mean()
    * 100
)
pct_historico = (df_filt["family_history"] == "yes").mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pacientes na base", f"{total_pacientes:,}".replace(",", "."))
k2.metric("IMC medio", f"{imc_medio:.2f}")
k3.metric("% com obesidade", f"{pct_obesidade:.1f}%")
k4.metric("% com historico familiar", f"{pct_historico:.1f}%")

st.divider()

# -----------------------------------------------------------------------------
# Distribuicao do alvo
# -----------------------------------------------------------------------------
st.subheader("Distribuicao da classe alvo")
class_counts = (
    df_filt["Obesity"].value_counts().reindex(ORDERED_CLASSES).fillna(0).reset_index()
)
class_counts.columns = ["Classe", "Quantidade"]
fig_classes = px.bar(
    class_counts,
    x="Classe",
    y="Quantidade",
    text="Quantidade",
    title="Quantidade de pacientes por nivel de obesidade",
    color="Classe",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
fig_classes.update_traces(textposition="outside")
fig_classes.update_layout(showlegend=False, xaxis_title="", yaxis_title="Pacientes")
st.plotly_chart(fig_classes, use_container_width=True)
st.caption(
    "O dataset eh relativamente equilibrado entre as classes, mas vale "
    "observar que pacientes com sobrepeso e obesidade somados representam "
    "a maior parte da amostra."
)

st.divider()

# -----------------------------------------------------------------------------
# IMC por classe e por genero
# -----------------------------------------------------------------------------
st.subheader("IMC, idade e genero por nivel de obesidade")
g1, g2 = st.columns(2)

with g1:
    fig_box_imc = px.box(
        df_filt,
        x="Obesity",
        y="imc",
        color="Gender",
        category_orders={"Obesity": ORDERED_CLASSES},
        title="Distribuicao de IMC por classe e genero",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_box_imc.update_layout(xaxis_title="", yaxis_title="IMC")
    st.plotly_chart(fig_box_imc, use_container_width=True)

with g2:
    fig_box_age = px.box(
        df_filt,
        x="Obesity",
        y="Age",
        color="Gender",
        category_orders={"Obesity": ORDERED_CLASSES},
        title="Distribuicao de idade por classe e genero",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_box_age.update_layout(xaxis_title="", yaxis_title="Idade")
    st.plotly_chart(fig_box_age, use_container_width=True)

st.caption(
    "O IMC eh fortemente correlacionado com a classe alvo, como esperado, "
    "ja que o calculo deriva de peso e altura. A distribuicao por idade "
    "mostra que adultos mais velhos tendem a aparecer com mais frequencia "
    "nas classes de obesidade."
)

st.divider()

# -----------------------------------------------------------------------------
# Habitos vs obesidade
# -----------------------------------------------------------------------------
st.subheader("Habitos de vida e nivel de obesidade")

h1, h2 = st.columns(2)

with h1:
    # 1. Dicionários de tradução
    obesity_labels = {
        "Insufficient_Weight": "Abaixo do Peso",
        "Normal_Weight": "Peso Normal",
        "Overweight_Level_I": "Sobrepeso I",
        "Overweight_Level_II": "Sobrepeso II",
        "Obesity_Type_I": "Obesidade Grau I",
        "Obesity_Type_II": "Obesidade Grau II",
        "Obesity_Type_III": "Obesidade Grau III"
    }

    faf_labels = {
        0: "0 (Nenhuma)",
        1: "1 a 2 dias",
        2: "3 a 4 dias",
        3: "5+ dias"
    }

    # 2. Preparação dos dados
    # Criamos uma cópia para não poluir o df_filt usado nos outros gráficos
    df_h1 = df_filt.copy()
    
    # Arredondamos e mapeamos os valores
    df_h1["Atividade_Fisica"] = df_h1["FAF"].round().astype(int).map(faf_labels)
    df_h1["Classificacao"] = df_h1["Obesity"].map(obesity_labels)

    # Agrupamento usando as categorias já limpas
    faf_df = df_h1.groupby(["Atividade_Fisica", "Classificacao"]).size().reset_index(name="Quantidade")

    # A lista de ordem usa a constante ORDERED_CLASSES já definida no topo do seu arquivo
    ordered_labels_pt = [obesity_labels[c] for c in ORDERED_CLASSES]

    # 3. Geração do gráfico
    fig_faf = px.bar(
        faf_df,
        x="Atividade_Fisica",
        y="Quantidade",
        color="Classificacao",
        text="Quantidade", # Adiciona os valores em cada barra
        category_orders={
            "Classificacao": ordered_labels_pt, 
            "Atividade_Fisica": list(faf_labels.values())
        },
        title="Frequência de Atividade Física vs Nível de Obesidade",
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    # 4. Ajustes de layout e formatação das barras
    fig_faf.update_layout(
        barmode="group", 
        xaxis_title="Frequência Semanal",
        yaxis_title="Número de Pacientes",
        legend_title="Categoria"
    )

    fig_faf.update_traces(
        textposition='auto',
        textfont_size=12,
        marker_line_width=0 # Deixa a barra "lisa"
    )

    # 5. Renderização no Streamlit
    st.plotly_chart(fig_faf, use_container_width=True)

with h2:
    # Proporcao de obesidade por consumo calorico frequente (FAVC).
    favc_df = (
        df_filt.groupby(["FAVC", "Obesity"]).size().reset_index(name="Quantidade")
    )
    fig_favc = px.bar(
        favc_df,
        x="FAVC",
        y="Quantidade",
        color="Obesity",
        category_orders={"Obesity": ORDERED_CLASSES},
        title="Consumo frequente de alimentos caloricos vs obesidade",
        labels={"FAVC": "Consome alimentos caloricos com frequencia?"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_favc.update_layout(barmode="stack", yaxis_title="Pacientes")
    st.plotly_chart(fig_favc, use_container_width=True)

st.caption(
    "Pacientes com baixa frequencia de atividade fisica e alto consumo de "
    "alimentos caloricos aparecem com mais peso nas classes de obesidade. "
    "Essa eh uma associacao classica e tambem se reflete no modelo treinado."
)

# Histograma de IMC por faixa.
fig_imc_faixa = px.histogram(
    df_filt,
    x="imc",
    color="IMC_Faixa",
    nbins=40,
    title="Histograma de IMC com faixas clinicas",
    color_discrete_sequence=px.colors.sequential.Sunsetdark,
)
fig_imc_faixa.update_layout(xaxis_title="IMC", yaxis_title="Pacientes")
st.plotly_chart(fig_imc_faixa, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Historico familiar e meio de transporte
# -----------------------------------------------------------------------------
st.subheader("Fatores de contexto: historico familiar e transporte")

f1, f2 = st.columns(2)

with f1:
    fh_df = (
        df_filt.groupby(["family_history", "Obesity"]).size().reset_index(name="Quantidade")
    )
    fig_fh = px.bar(
        fh_df,
        x="family_history",
        y="Quantidade",
        color="Obesity",
        category_orders={"Obesity": ORDERED_CLASSES},
        title="Historico familiar de sobrepeso vs nivel de obesidade",
        labels={"family_history": "Historico familiar?"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_fh.update_layout(barmode="stack", yaxis_title="Pacientes")
    st.plotly_chart(fig_fh, use_container_width=True)

with f2:
    mt_df = (
        df_filt.groupby(["MTRANS", "Obesity"]).size().reset_index(name="Quantidade")
    )
    fig_mt = px.bar(
        mt_df,
        x="MTRANS",
        y="Quantidade",
        color="Obesity",
        category_orders={"Obesity": ORDERED_CLASSES},
        title="Meio de transporte habitual vs nivel de obesidade",
        labels={"MTRANS": "Meio de transporte"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_mt.update_layout(barmode="stack", yaxis_title="Pacientes")
    st.plotly_chart(fig_mt, use_container_width=True)

st.caption(
    "O historico familiar aparece em alta proporcao nas faixas mais elevadas "
    "de obesidade, reforcando o fator genetico. O meio de transporte ajuda "
    "a indicar o nivel de movimentacao no dia a dia: deslocamentos a pe e "
    "de bicicleta sao menos frequentes nas classes mais graves."
)

st.divider()

# -----------------------------------------------------------------------------
# Correlacao numerica
# -----------------------------------------------------------------------------
st.subheader("Correlacao entre variaveis numericas")
num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "imc",
            "Healthy_Score", "Sedentary_Index"]
corr = df_filt[num_cols].corr().round(2)
fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Matriz de correlacao",
)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption(
    "IMC e peso tem correlacao alta (esperado). FAF e Sedentary_Index "
    "apresentam relacao inversa, e o Healthy_Score capta uma combinacao "
    "positiva entre vegetais, hidratacao e atividade fisica."
)

st.divider()

# -----------------------------------------------------------------------------
# Qualidade do modelo
# -----------------------------------------------------------------------------
st.subheader("Qualidade do modelo treinado")
if metrics is None:
    st.info(
        "Relatorio de metricas indisponivel. Execute `python train_model.py` "
        "para gerar `models/metrics_report.json`."
    )
else:
    st.markdown(
        f"**Modelo escolhido:** `{metrics['best_model']}`  \n"
        f"**F1-macro no holdout:** `{metrics['best_test_f1_macro']:.4f}`"
    )

    # Tabela comparativa entre os modelos avaliados.
    rows = []
    for model_name, info in metrics["results_by_model"].items():
        rows.append(
            {
                "Modelo": model_name,
                "CV F1-macro": round(info["cv_f1_macro"], 4),
                "Test Accuracy": round(info["test_accuracy"], 4),
                "Test F1-macro": round(info["test_f1_macro"], 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Matriz de confusao do modelo escolhido.
    best_cm = metrics["results_by_model"][metrics["best_model"]]["confusion_matrix"]
    cm_df = pd.DataFrame(best_cm, index=TARGET_CLASSES, columns=TARGET_CLASSES)
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Matriz de confusao - {metrics['best_model']}",
        labels={"x": "Predito", "y": "Real"},
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        "A matriz de confusao mostra que o modelo erra raramente, e quando "
        "erra, geralmente confunde classes adjacentes (por exemplo Sobrepeso "
        "I com Sobrepeso II). Esse padrao reduz o risco de erros graves."
    )
