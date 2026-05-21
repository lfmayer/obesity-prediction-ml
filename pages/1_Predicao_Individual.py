# =============================================================================
# pages/1_Predicao_Individual.py
# -----------------------------------------------------------------------------
# Pagina de predicao individual. Recebe os dados de uma pessoa via formulario,
# aplica as mesmas transformacoes usadas no treino (importadas de
# train_model.py para evitar divergencia) e exibe o nivel de obesidade
# previsto junto com a confianca por classe.
# =============================================================================

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Adiciona o diretorio raiz do projeto ao sys.path para conseguir importar
# train_model.py (que vive um nivel acima desta pagina). Esse truque eh
# necessario porque o Streamlit roda cada pagina como modulo independente.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_model import (  # noqa: E402  (import depois do sys.path eh proposital)
    FEATURE_COLUMNS,
    MODEL_PATH,
    add_engineered_features,
    encode_categoricals,
)

# -----------------------------------------------------------------------------
# Configuracao da pagina
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicao Individual",
    page_icon="🍽️",
    layout="centered",
)

st.title("🍽️ Predicao de Nivel de Obesidade")
st.markdown(
    "Preencha os dados abaixo e clique em **Prever Nivel de Obesidade**. "
    "O modelo retorna a classe estimada e a confianca de cada uma das sete "
    "categorias possiveis."
)


# -----------------------------------------------------------------------------
# Carregamento do modelo
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Carrega o pipeline serializado em disco. O cache_resource garante
    que o modelo seja carregado apenas uma vez por sessao do Streamlit."""
    if not MODEL_PATH.exists():
        st.error(
            "Arquivo do modelo nao encontrado. "
            "Execute `python train_model.py` na raiz do projeto antes de usar a aplicacao."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()
st.success("Modelo carregado com sucesso.")

# Dicionario para apresentar os nomes das classes de forma amigavel ao
# usuario (em portugues e com emoji indicando a gravidade).
TRANSLATION_DICT = {
    "Insufficient_Weight": "Peso Insuficiente 🦴",
    "Normal_Weight": "Peso Normal ✅",
    "Overweight_Level_I": "Sobrepeso Grau I 📈",
    "Overweight_Level_II": "Sobrepeso Grau II 📊",
    "Obesity_Type_I": "Obesidade Grau I ⚠️",
    "Obesity_Type_II": "Obesidade Grau II 🚨",
    "Obesity_Type_III": "Obesidade Grau III 🔴",
}

# -----------------------------------------------------------------------------
# Formulario de entrada
# -----------------------------------------------------------------------------
with st.form("form_predicao"):
    st.subheader("📋 Dados Pessoais")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Genero", ["Female", "Male"])
        age = st.slider("Idade (anos)", min_value=14, max_value=80, value=25, step=1)
        height = st.number_input(
            "Altura (m)", min_value=1.0, max_value=2.2, value=1.70, step=0.01, format="%.2f"
        )
        weight = st.number_input(
            "Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5, format="%.1f"
        )
    with col2:
        family_history = st.selectbox("Historico familiar de sobrepeso", ["yes", "no"])
        favc = st.selectbox("Consome alimentos caloricos com frequencia?", ["yes", "no"])
        fcvc = st.slider(
            "Consumo de vegetais (1 = raramente, 2 = as vezes, 3 = sempre)", 1, 3, 2
        )
        ncp = st.slider("Numero de refeicoes principais por dia", 1, 4, 3)

    st.subheader("🏃 Habitos e Estilo de Vida")
    col3, col4 = st.columns(2)
    with col3:
        caec = st.selectbox(
            "Consome comida entre refeicoes?",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        smoke = st.selectbox("Fumante?", ["yes", "no"])
        ch2o = st.slider("Consumo de agua (1 = <1L, 2 = 1-2L, 3 = >2L)", 1, 3, 2)
        scc = st.selectbox("Monitora consumo calorico?", ["yes", "no"])
    with col4:
        faf = st.slider(
            "Atividade fisica semanal (0 = nenhuma, 1 = 1-2x, 2 = 3-4x, 3 = 5x+)",
            0,
            3,
            1,
        )
        tue = st.slider(
            "Tempo em dispositivos eletronicos (0 = 0-2h, 1 = 3-5h, 2 = >5h)", 0, 2, 1
        )
        calc = st.selectbox(
            "Consome alcool?",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        mtrans = st.selectbox(
            "Meio de transporte principal",
            ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"],
        )

    st.caption(
        "IMC, Healthy Score e Indice de Sedentarismo sao calculados "
        "automaticamente a partir dos dados informados."
    )

    submitted = st.form_submit_button("🔍 Prever Nivel de Obesidade")


# -----------------------------------------------------------------------------
# Predicao
# -----------------------------------------------------------------------------
if submitted:
    with st.spinner("Processando dados e realizando predicao..."):
        # Monta um DataFrame com as colunas brutas, exatamente no formato que
        # o pipeline de treino espera. Em seguida, aplica as mesmas funcoes
        # de feature engineering importadas de train_model.py.
        raw_df = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Age": age,
                    "Height": height,
                    "Weight": weight,
                    "family_history": family_history,
                    "FAVC": favc,
                    "FCVC": fcvc,
                    "NCP": ncp,
                    "CAEC": caec,
                    "SMOKE": smoke,
                    "CH2O": ch2o,
                    "SCC": scc,
                    "FAF": faf,
                    "TUE": tue,
                    "CALC": calc,
                    "MTRANS": mtrans,
                }
            ]
        )

        # Aplica feature engineering e encodings na mesma ordem do treino.
        feat_df = add_engineered_features(raw_df)
        feat_df = encode_categoricals(feat_df)

        # Seleciona apenas as colunas que entram no modelo. Reindexar protege
        # contra qualquer mudanca acidental de ordem.
        X_input = feat_df.reindex(columns=FEATURE_COLUMNS)

        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]

        # Recupera a ordem das classes diretamente do modelo (assim nao
        # dependemos de constantes externas para manter a sincronia).
        class_names = list(model.classes_)
        proba_dict = {
            cls: float(prob) for cls, prob in zip(class_names, prediction_proba)
        }

    # Bloco de exibicao do resultado.
    st.write("---")
    st.markdown("### 🧬 Resultado do Modelo:")
    result_display = TRANSLATION_DICT.get(prediction, prediction)
    st.success(f"## Nivel Classificado: **{result_display}**")

    # Mensagens contextuais empaticas, respeitando a sensibilidade do tema.
    if prediction == "Normal_Weight":
        st.balloons()
        st.info(
            "💡 Parabens! Os habitos atuais estao ajudando a manter o peso em "
            "uma faixa saudavel. Continue mantendo a rotina."
        )
    elif prediction == "Insufficient_Weight":
        st.warning(
            "⚠️ Atencao: o modelo indicou peso insuficiente. Eh recomendavel "
            "consultar um nutricionista para avaliar a ingestao calorica e "
            "de nutrientes de forma adequada."
        )
    elif "Overweight" in prediction:
        st.warning(
            "📊 Atencao: o modelo identificou um nivel de sobrepeso. Pequenos "
            "ajustes na alimentacao e a pratica regular de exercicios podem "
            "ajudar a reverter esse quadro."
        )
    elif "Obesity" in prediction:
        st.error(
            "🚨 Alerta de Saude: o modelo classificou o perfil em uma faixa de "
            "obesidade. A obesidade eh uma condicao medica complexa, e o "
            "acompanhamento por equipe de saude (medicos e nutricionistas) "
            "eh o caminho mais seguro."
        )

    # Tabela com a confianca por classe. Util para o profissional de saude
    # ver se a decisao do modelo foi clara ou se ficou entre duas categorias.
    with st.expander("🔍 Detalhes Tecnicos (Probabilidade por Classe)"):
        st.write("Probabilidades calculadas pelo modelo para cada faixa:")
        translated_proba = {
            TRANSLATION_DICT.get(cls, cls): f"{prob * 100:.2f}%"
            for cls, prob in proba_dict.items()
        }
        df_proba = pd.DataFrame(
            translated_proba.items(), columns=["Classificacao", "Confianca do Modelo"]
        )
        st.dataframe(df_proba, hide_index=True, use_container_width=True)

    # Resumo das features derivadas para transparencia clinica.
    with st.expander("📐 Features derivadas calculadas para esta predicao"):
        imc_value = X_input.iloc[0]["imc"]
        healthy_value = X_input.iloc[0]["Healthy_Score"]
        sedentary_value = X_input.iloc[0]["Sedentary_Index"]
        c1, c2, c3 = st.columns(3)
        c1.metric("IMC", f"{imc_value:.2f}")
        c2.metric("Healthy Score", f"{healthy_value:.2f}")
        c3.metric("Indice de Sedentarismo", f"{sedentary_value:.2f}")
        st.caption(
            "IMC = Peso / (Altura ** 2). Healthy Score = FCVC + FAF + CH2O. "
            "Indice de Sedentarismo = TUE / (FAF + 1)."
        )
