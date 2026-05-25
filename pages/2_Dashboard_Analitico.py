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
    
    # 1. Dicionários rápidos para traduzir a interface dos filtros
    gender_labels = {"Female": "Feminino", "Male": "Masculino"}
    fh_labels = {"yes": "Sim", "no": "Não"}

    # 2. Aplicando o format_func para mudar apenas a exibição
    selected_genders = st.multiselect(
        "Gênero",
        options=sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique()),
        format_func=lambda x: gender_labels.get(x, x) # <--- O truque aqui!
    )
    
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider(
        "Faixa de idade",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )
    
    selected_history = st.multiselect(
        "Histórico familiar de sobrepeso",
        options=sorted(df["family_history"].unique()),
        default=sorted(df["family_history"].unique()),
        format_func=lambda x: fh_labels.get(x, x) # <--- Aqui também!
    )
    
    selected_classes = st.multiselect(
        "Níveis de obesidade",
        options=ORDERED_CLASSES,
        default=ORDERED_CLASSES,
        format_func=lambda x: obesity_labels.get(x, x) # <--- Usando o dicionário global
    )
    
    st.caption(
        "Os filtros se aplicam a todos os gráficos abaixo. Limpe um filtro "
        "para voltar à base completa."
    )


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
st.subheader("Distribuição da classe alvo")

# 1. Conta os valores mantendo a estrutura original
class_counts = (
    df_filt["Obesity"].value_counts().reindex(ORDERED_CLASSES).fillna(0).reset_index()
)
class_counts.columns = ["Classe", "Quantidade"]

# 2. Traduz as classes para o português usando o dicionário global
class_counts["Classe"] = class_counts["Classe"].map(obesity_labels)

fig_classes = px.bar(
    class_counts,
    x="Classe",
    y="Quantidade",
    text="Quantidade",
    title="Quantidade de pacientes por nível de obesidade",
    color="Classe",
    category_orders={"Classe": ordered_labels_pt}, # Garante a ordem visual correta
    color_discrete_sequence=px.colors.sequential.Viridis,
)

fig_classes.update_traces(textposition="outside", marker_line_width=0)

# 3. Ajuste do Layout: bargap para afinar as barras
fig_classes.update_layout(
    showlegend=False, 
    xaxis_title="", 
    yaxis_title="Pacientes",
    bargap=0.4  # <-- Aumenta o espaço entre barras, deixando-as mais finas
)

st.plotly_chart(fig_classes, use_container_width=True)

st.caption(
    "O dataset é relativamente equilibrado entre as classes, mas vale "
    "observar que pacientes com sobrepeso e obesidade somados representam "
    "a maior parte da amostra."
)

# -----------------------------------------------------------------------------
# IMC por classe e por genero
# -----------------------------------------------------------------------------
st.subheader("IMC, idade e gênero por nível de obesidade")

# 1. Preparar os dados para este gráfico (tradução)
df_box = df_filt.copy()

# Usamos o dicionário global para traduzir o eixo X
df_box["Classificacao"] = df_box["Obesity"].map(obesity_labels)

# Traduzindo a legenda de Gênero
gender_labels = {"Female": "Feminino", "Male": "Masculino"}
df_box["Genero"] = df_box["Gender"].map(gender_labels)

# 2. Selecionar cores de alto contraste dentro da Viridis
cores_viridis_contraste = [px.colors.sequential.Viridis[0], px.colors.sequential.Viridis[6]]

g1, g2 = st.columns(2)

with g1:
    fig_box_imc = px.box(
        df_box,
        x="Classificacao",
        y="imc",
        color="Genero",
        category_orders={"Classificacao": ordered_labels_pt},
        title="Distribuição de IMC por classe e gênero",
        color_discrete_sequence=cores_viridis_contraste,
    )
    fig_box_imc.update_layout(xaxis_title="", yaxis_title="IMC", legend_title="Gênero")
    st.plotly_chart(fig_box_imc, use_container_width=True)

with g2:
    fig_box_age = px.box(
        df_box,
        x="Classificacao",
        y="Age",
        color="Genero",
        category_orders={"Classificacao": ordered_labels_pt},
        title="Distribuição de Idade por classe e gênero",
        color_discrete_sequence=cores_viridis_contraste,
    )
    fig_box_age.update_layout(xaxis_title="", yaxis_title="Idade", legend_title="Gênero")
    st.plotly_chart(fig_box_age, use_container_width=True)

st.caption(
    "O IMC é fortemente correlacionado com a classe alvo, como esperado, "
    "já que o cálculo deriva de peso e altura. A distribuição por idade "
    "mostra que adultos mais velhos tendem a aparecer com mais frequência "
    "nas classes de obesidade."
)

st.divider()

# -----------------------------------------------------------------------------
# Habitos vs obesidade
# -----------------------------------------------------------------------------
st.subheader("Habitos de vida e nivel de obesidade")

# --- Gráfico de Atividade Física (Ocupando a tela inteira) ---
df_faf = df_filt.copy()
faf_labels = {
    0: "0 (Nenhuma)",
    1: "1 a 2 dias",
    2: "3 a 4 dias",
    3: "5+ dias"
}

df_faf["Atividade_Fisica"] = df_faf["FAF"].round().astype(int).map(faf_labels)
df_faf["Classificacao"] = df_faf["Obesity"].map(obesity_labels)

faf_grouped = df_faf.groupby(["Atividade_Fisica", "Classificacao"]).size().reset_index(name="Quantidade")

fig_faf = px.bar(
    faf_grouped,
    x="Atividade_Fisica",
    y="Quantidade",
    color="Classificacao",
    text="Quantidade",
    category_orders={
        "Classificacao": ordered_labels_pt, 
        "Atividade_Fisica": list(faf_labels.values())
    },
    title="Frequência de Atividade Física vs Nível de Obesidade",
    color_discrete_sequence=px.colors.sequential.Viridis
)

fig_faf.update_layout(
    barmode="group", 
    xaxis_title="Frequência Semanal",
    yaxis_title="Número de Pacientes",
    legend_title="Categoria"
)
fig_faf.update_traces(textposition='auto', textfont_size=12, marker_line_width=0)
st.plotly_chart(fig_faf, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Gráfico de Consumo Calórico (Ocupando a tela inteira) ---
df_favc = df_filt.copy()
favc_labels = {"yes": "Sim", "no": "Não"}

df_favc["Consumo_Calorico"] = df_favc["FAVC"].map(favc_labels)
df_favc["Classificacao"] = df_favc["Obesity"].map(obesity_labels)

favc_grouped = df_favc.groupby(["Consumo_Calorico", "Classificacao"]).size().reset_index(name="Quantidade")

fig_favc = px.bar(
    favc_grouped,
    x="Consumo_Calorico",
    y="Quantidade",
    color="Classificacao",
    text="Quantidade",
    category_orders={
        "Classificacao": ordered_labels_pt, 
        "Consumo_Calorico": ["Não", "Sim"]
    },
    title="Consumo frequente de alimentos calóricos vs Nível de Obesidade",
    color_discrete_sequence=px.colors.sequential.Viridis
)

fig_favc.update_layout(
    barmode="group", 
    xaxis_title="Consumo Frequente?",
    yaxis_title="Número de Pacientes",
    legend_title="Categoria"
)
fig_favc.update_traces(textposition='auto', textfont_size=12, marker_line_width=0)
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
    color_discrete_sequence=px.colors.sequential.Viridis,
)
fig_imc_faixa.update_layout(xaxis_title="IMC", yaxis_title="Pacientes")
st.plotly_chart(fig_imc_faixa, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Historico familiar e meio de transporte
# -----------------------------------------------------------------------------
st.subheader("Fatores de contexto: histórico familiar e transporte")

# --- Gráfico de Histórico Familiar (Ocupando a tela inteira) ---
df_fh = df_filt.copy()
fh_labels = {"yes": "Sim", "no": "Não"}

# Traduzindo os dados
df_fh["Historico_Familiar"] = df_fh["family_history"].map(fh_labels)
df_fh["Classificacao"] = df_fh["Obesity"].map(obesity_labels)

# Agrupando os dados limpos
fh_grouped = df_fh.groupby(["Historico_Familiar", "Classificacao"]).size().reset_index(name="Quantidade")

fig_fh = px.bar(
    fh_grouped,
    x="Historico_Familiar",
    y="Quantidade",
    color="Classificacao",
    text="Quantidade", # Adiciona os rótulos numéricos
    category_orders={
        "Classificacao": ordered_labels_pt,
        "Historico_Familiar": ["Não", "Sim"]
    },
    title="Histórico familiar de sobrepeso vs Nível de Obesidade",
    color_discrete_sequence=px.colors.sequential.Viridis
)

fig_fh.update_layout(
    barmode="group", 
    xaxis_title="Possui Histórico Familiar?",
    yaxis_title="Número de Pacientes",
    legend_title="Categoria"
)
fig_fh.update_traces(textposition='auto', textfont_size=12, marker_line_width=0)
st.plotly_chart(fig_fh, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True) # Adiciona um respiro visual entre os gráficos

# --- Gráfico de Meio de Transporte (Ocupando a tela inteira) ---
df_mt = df_filt.copy()
mt_labels = {
    "Automobile": "Automóvel",
    "Public_Transportation": "Transporte Público",
    "Motorbike": "Moto",
    "Bike": "Bicicleta",
    "Walking": "A pé"
}

# Traduzindo os dados
df_mt["Meio_Transporte"] = df_mt["MTRANS"].map(mt_labels)
df_mt["Classificacao"] = df_mt["Obesity"].map(obesity_labels)

mt_grouped = df_mt.groupby(["Meio_Transporte", "Classificacao"]).size().reset_index(name="Quantidade")

# Definindo uma ordem lógica para o eixo X (dos motorizados para os ativos)
ordered_mtrans = ["Automóvel", "Transporte Público", "Moto", "Bicicleta", "A pé"]

fig_mt = px.bar(
    mt_grouped,
    x="Meio_Transporte",
    y="Quantidade",
    color="Classificacao",
    text="Quantidade", # Adiciona os rótulos numéricos
    category_orders={
        "Classificacao": ordered_labels_pt,
        "Meio_Transporte": ordered_mtrans
    },
    title="Meio de transporte habitual vs Nível de Obesidade",
    color_discrete_sequence=px.colors.sequential.Viridis
)

fig_mt.update_layout(
    barmode="group", 
    xaxis_title="Meio de Transporte Habitual",
    yaxis_title="Número de Pacientes",
    legend_title="Categoria"
)
fig_mt.update_traces(textposition='auto', textfont_size=12, marker_line_width=0)
st.plotly_chart(fig_mt, use_container_width=True)

st.caption(
    "O histórico familiar aparece em alta proporção nas faixas mais elevadas "
    "de obesidade, reforçando o fator genético. O meio de transporte ajuda "
    "a indicar o nível de movimentação no dia a dia: deslocamentos a pé e "
    "de bicicleta são menos frequentes nas classes mais graves."
)
# -----------------------------------------------------------------------------
# Correlacao numerica
# -----------------------------------------------------------------------------
st.subheader("Correlação entre variáveis numéricas")

num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "imc", "Healthy_Score", "Sedentary_Index"]

# 1. Dicionário para traduzir os eixos do gráfico para o usuário
col_labels = {
    "Age": "Idade",
    "Height": "Altura",
    "Weight": "Peso",
    "FCVC": "Consumo de Vegetais",
    "NCP": "Refeições por dia",
    "CH2O": "Água diária",
    "FAF": "Atividade Física",
    "TUE": "Tempo em Telas",
    "imc": "IMC",
    "Healthy_Score": "Score Saudável",
    "Sedentary_Index": "Índice Sedentário"
}

# Calcula a correlação
corr = df_filt[num_cols].corr().round(2)

# 2. Aplica a tradução nas linhas (index) e colunas
corr = corr.rename(columns=col_labels, index=col_labels)

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Viridis",
    title="Matriz de Correlação",
    aspect="auto" # <-- Libera o Plotly para esticar os quadrados
)

# 3. O "Pulo do Gato": Aumentar a altura e ajustar as margens
fig_corr.update_layout(
    height=800, # <-- Define uma altura imponente para a matriz (pode aumentar se quiser)
    margin=dict(l=0, r=0, t=50, b=0) # Remove bordas brancas excessivas
)

st.plotly_chart(fig_corr, use_container_width=True)

st.caption(
    "IMC e peso têm correlação alta (esperado). Atividade Física e Índice Sedentário "
    "apresentam relação inversa, e o Score Saudável capta uma combinação "
    "positiva entre vegetais, hidratação e atividade física."
)

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
    
    # 1. Traduzindo as classes usando o dicionário global que já criamos
    translated_classes = [obesity_labels.get(c, c) for c in TARGET_CLASSES]
    
    cm_df = pd.DataFrame(best_cm, index=translated_classes, columns=translated_classes)
    
    # 2. Gerando o gráfico com paleta 'Blues' e aspect='auto'
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues", # Deixa o fundo (zeros) branco/claro
        title=f"Matriz de Confusão - {metrics['best_model']}",
        labels={"x": "Predito", "y": "Real"},
        aspect="auto" # Permite esticar a matriz
    )
    
    # 3. Ajustando a altura para o gráfico ficar imponente
    fig_cm.update_layout(
        height=700, 
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        "A matriz de confusão mostra que o modelo erra raramente, e quando "
        "erra, geralmente confunde classes adjacentes (por exemplo Sobrepeso "
        "I com Sobrepeso II). Esse padrão reduz o risco de erros graves."
    )
