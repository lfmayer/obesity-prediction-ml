# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# Pagina inicial da aplicacao Streamlit. Atua como apresentacao do projeto
# e ponto de entrada para as duas paginas funcionais (preditivo e dashboard).
#
# A organizacao multipagina segue a convencao do Streamlit:
#   - app.py             -> pagina inicial (esta)
#   - pages/1_*.py       -> pagina de predicao individual
#   - pages/2_*.py       -> pagina de dashboard analitico
# Esses arquivos sao detectados automaticamente pelo Streamlit em runtime.
# =============================================================================

import streamlit as st

# Configuracao geral da pagina (titulo na aba do navegador, layout, etc).
st.set_page_config(
    page_title="Predicao de Obesidade - Tech Challenge FIAP",
    page_icon="🏥",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Conteudo da pagina inicial
# -----------------------------------------------------------------------------
st.title("🏥 Predicao de Nivel de Obesidade")
st.subheader("Tech Challenge - FIAP - Fase 4 - Data Analytics")

st.markdown(
    """
    Esta aplicacao foi desenvolvida como entregavel do Tech Challenge da Fase 4
    e tem dois objetivos principais:

    1. **Aplicar um modelo preditivo** que classifica o nivel de obesidade de
       uma pessoa a partir de dados demograficos e de habitos de vida.
    2. **Apresentar um painel analitico** com os principais insights extraidos
       da base de dados, voltado para a tomada de decisao da equipe medica.

    A base de dados utilizada (`Obesity.csv`) contem 2.111 registros com
    variaveis sobre genero, idade, peso, altura, historico familiar, alimentacao,
    atividade fisica e meio de transporte. O alvo eh uma classe categorica com
    sete niveis, indo de "Insufficient_Weight" ate "Obesity_Type_III".
    """
)

# Cards de navegacao para guiar o usuario ate as paginas funcionais.
col1, col2 = st.columns(2)
with col1:
    st.info(
        "**Pagina 1 - Predicao Individual**\n\n"
        "Use o menu lateral esquerdo para abrir a pagina *Predicao Individual*. "
        "La voce informa os dados de uma pessoa e o modelo retorna a "
        "classificacao do nivel de obesidade junto com a confianca de cada classe."
    )
with col2:
    st.info(
        "**Pagina 2 - Dashboard Analitico**\n\n"
        "Use o menu lateral esquerdo para abrir o *Dashboard Analitico*. "
        "Ele traz visualizacoes de distribuicao das classes, IMC, habitos "
        "alimentares e atividade fisica para apoiar a leitura clinica."
    )

st.divider()

# Resumo tecnico do modelo escolhido. Mantemos visivel para que a equipe
# medica saiba qual algoritmo esta por tras das predicoes.
st.markdown(
    """
    ### Sobre o modelo

    O modelo final foi escolhido apos uma comparacao entre tres algoritmos com
    busca de hiperparametros (`GridSearchCV` com validacao cruzada
    estratificada de 5 folds):

    - **Logistic Regression** (modelo linear de referencia)
    - **Random Forest** (ensemble por bagging)
    - **Gradient Boosting** (ensemble por boosting)

    A metrica usada para selecao foi **F1 macro**, escolhida por dar peso
    igual a todas as classes mesmo quando ha pequeno desbalanceamento. O
    pipeline completo (pre-processamento + modelo) esta serializado em
    `models/best_pipeline.joblib` e o relatorio de metricas em
    `models/metrics_report.json`.

    O codigo de treino fica em `train_model.py` e pode ser reexecutado a
    qualquer momento para regenerar o modelo.
    """
)

st.caption(
    "Aplicacao desenvolvida com Streamlit. Codigo aberto no repositorio do "
    "Tech Challenge."
)
