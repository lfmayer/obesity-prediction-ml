# Previsão de Risco de Obesidade — Sistema de Machine Learning End-to-End

Um sistema de classificação multi-classe supervisionado que identifica níveis de risco de obesidade em **7 categorias clínicas** com base em hábitos alimentares, atividade física e dados de estilo de vida. O pipeline treinado é disponibilizado por meio de uma aplicação web interativa sem código, construída com Streamlit.

**Demo ao vivo:** https://fiap-tc4-obesidade-jdhk9vfrpcjj9g5uzwh6xg.streamlit.app/

---

## Definição do Problema

A obesidade é uma condição crônica complexa e multifatorial associada ao diabetes tipo 2, doenças cardiovasculares e maior mortalidade por todas as causas. As equipes de saúde frequentemente carecem de ferramentas escaláveis e orientadas por dados para estratificar pacientes por nível de risco antes de uma avaliação clínica formal — o que leva a intervenções tardias e triagem inconsistente entre profissionais.

## Solução

Um pipeline de ML reproduzível que:
1. Limpa dados brutos de pesquisa e cria features com base em domínio clínico (IMC, Índice de Saúde, Índice Sedentário)
2. Compara três algoritmos de classificação usando validação cruzada estratificada
3. Seleciona automaticamente o melhor pipeline com base no F1-macro médio
4. Expõe previsões por meio de um app Streamlit de duas páginas — formulário de predição individual e painel analítico — sem necessidade de codificação pelos usuários finais

## Resultados

| Modelo | F1-macro (CV) | Acurácia (Teste) | F1-macro (Teste) |
|---|---|---|---|
| Regressão Logística | 0,942 | 94,0% | 0,938 |
| Random Forest | 0,976 | 96,9% | 0,967 |
| **Gradient Boosting** ✓ | **0,971** | **97,1%** | **0,970** |

**Meta base do projeto:** 75% de acurácia · **Alcançado:** 97,1%

O modelo vencedor (Gradient Boosting) classifica erroneamente apenas classes adjacentes em casos limítrofes ambíguos — por exemplo, Sobrepeso Nível I vs. II — o que minimiza o risco clínico de um diagnóstico equivocado grave.

## Engenharia de Features

Três features compostas foram derivadas das variáveis brutas da pesquisa antes do treinamento:

| Feature | Fórmula | Justificativa |
|---|---|---|
| **IMC** | `Peso / Altura²` | Indicador clínico padrão de composição corporal |
| **Índice de Saúde** | `Consumo de vegetais + Atividade física + Água diária` | Agrega comportamentos de saúde positivos em um único proxy |
| **Índice Sedentário** | `Tempo de tela / (Atividade física + 1)` | Captura a razão entre comportamento sedentário e ativo; `+1` evita divisão por zero |

Variáveis categóricas ordinais (frequência de lanches, consumo de álcool, modal de transporte) foram mapeadas manualmente para escalas inteiras que preservam sua ordenação natural de intensidade, em vez de aplicar one-hot encoding, mantendo o espaço de features compacto e interpretável.

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Processamento de dados | pandas, NumPy |
| Pipeline de ML | scikit-learn (Pipeline, ColumnTransformer, GridSearchCV) |
| Modelos avaliados | LogisticRegression, RandomForestClassifier, GradientBoostingClassifier |
| Serialização | joblib |
| Aplicação web | Streamlit |
| Visualizações | Plotly Express |
| Deploy | Streamlit Community Cloud |

## Páginas da Aplicação

**Página 1 — Predição Individual**
Uma interface baseada em formulário onde o profissional de saúde preenche os dados demográficos e de estilo de vida do paciente. O app aplica o mesmo pré-processamento utilizado no treinamento (importado diretamente de `train_model.py` para evitar discrepância entre treino e produção), e retorna:
- A classe de obesidade prevista
- Probabilidades de confiança por classe
- As features derivadas computadas (IMC, Índice de Saúde, Índice Sedentário) para transparência

**Página 2 — Painel Analítico**
Uma visão exploratória interativa do dataset completo com filtros na barra lateral (gênero, faixa etária, histórico familiar, nível de obesidade). Inclui:
- Cards de KPI resumidos (contagem de pacientes, IMC médio, prevalência de obesidade, taxa de histórico familiar)
- Gráfico de barras com distribuição das classes alvo
- Boxplots de IMC e idade segmentados por classe e gênero
- Frequência de atividade física e hábitos de consumo calórico vs. nível de obesidade
- Tabulações cruzadas de histórico familiar e modal de transporte
- Mapa de calor de correlação numérica
- Tabela de comparação de modelos e matriz de confusão para o melhor modelo selecionado

## Estrutura do Projeto

```
obesity-prediction-ml/
├── app.py                        # Ponto de entrada do Streamlit (página inicial)
├── train_model.py                # Pipeline de treinamento completo — carregamento de dados,
│                                 #   engenharia de features, GridSearchCV, persistência do modelo
├── pages/
│   ├── 1_Predicao_Individual.py  # Formulário de predição individual
│   └── 2_Dashboard_Analitico.py  # Painel analítico
├── notebooks/
│   └── TechChallenge4_v2.ipynb   # EDA, justificativa de modelagem e reprodutibilidade
├── models/
│   ├── best_pipeline.joblib      # Pipeline sklearn serializado (pré-processador + modelo)
│   └── metrics_report.json       # Resultados completos de avaliação para todos os modelos candidatos
├── data/
│   └── Obesity.csv               # Dataset bruto (2.111 registros, 17 colunas)
├── requirements.txt
└── runtime.txt
```

## Dataset

- **Fonte:** UCI Machine Learning Repository — Estimativa dos Níveis de Obesidade com Base em Hábitos Alimentares e Condições Físicas
- **Registros:** 2.111 indivíduos do México, Peru e Colômbia
- **Alvo:** `Obesity` — 7 classes ordinais variando de `Insufficient_Weight` a `Obesity_Type_III`
- **Features de entrada:** Gênero, Idade, Altura, Peso, histórico familiar de sobrepeso, hábitos alimentares (consumo de vegetais, consumo de alimentos calóricos, refeições por dia, lanches, ingestão de água, álcool), atividade física, tempo de tela, monitoramento de calorias e modal de transporte

## Reprodutibilidade

A execução completa do treinamento pode ser refeita a qualquer momento:

```bash
python train_model.py
```

Isso regenera `models/best_pipeline.joblib` e `models/metrics_report.json` do zero. A semente aleatória é fixada (`random_state=42`) e a divisão treino/teste é estratificada (80/20), tornando os resultados determinísticos.

## Como Executar Localmente

```bash
git clone https://github.com/<seu-usuario>/obesity-prediction-ml.git
cd obesity-prediction-ml
pip install -r requirements.txt
streamlit run app.py
```

O app abrirá em `http://localhost:8501`. O modelo serializado já está incluído no repositório em `models/`, portanto nenhum retreinamento é necessário para usar a página de predição.

---

**FIAP PosTech** · Business Analytics & Data-Driven Decision Making  
Tech Challenge Fase 4 · Grupo 87
