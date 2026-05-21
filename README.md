# Tech Challenge - Fase 4 - Predicao de Obesidade

Projeto de Machine Learning desenvolvido como entregavel do Tech Challenge
da Fase 4 da Pos-Graduacao FIAP. O sistema classifica o nivel de obesidade
de um paciente em sete categorias e oferece um painel analitico para a
equipe medica.

Esta pasta (`desenvolvimento-gustavo/`) eh autocontida: tem dados, codigo,
modelo treinado, notebook e tudo o que eh necessario para rodar localmente
ou publicar online.

## Sumario

1. [O que tem na pasta](#o-que-tem-na-pasta)
2. [Como rodar localmente](#como-rodar-localmente)
3. [Como publicar no Streamlit Community Cloud](#como-publicar-no-streamlit-community-cloud)
4. [Como subir para o GitHub](#como-subir-para-o-github)
5. [Modelagem e resultados](#modelagem-e-resultados)
6. [Feature engineering](#feature-engineering)
7. [Como reabrir e re-executar o notebook](#como-reabrir-e-re-executar-o-notebook)
8. [Checklist de entrega](#checklist-de-entrega)
9. [Solucao de problemas](#solucao-de-problemas)

## O que tem na pasta

```
desenvolvimento-gustavo/
  app.py                            Pagina inicial da aplicacao Streamlit
  pages/
    1_Predicao_Individual.py        Formulario + predicao individual
    2_Dashboard_Analitico.py        Painel analitico interativo (Plotly)
  train_model.py                    Script que treina e seleciona o modelo
  notebooks/
    TechChallenge4_v2.ipynb         Notebook completo (EDA + modelagem)
  data/
    Obesity.csv                     Base de dados original
  models/
    best_pipeline.joblib            Modelo final serializado
    metrics_report.json             Relatorio de metricas dos 3 modelos
  requirements.txt                  Dependencias com versoes fixadas
  runtime.txt                       Versao do Python para o Streamlit Cloud
  .streamlit/config.toml            Tema e ajustes do app
  .gitignore                        Arquivos ignorados pelo Git
  README.md                         Este arquivo
  entrega.txt                       Template para os links finais da entrega
  venv/                             Ambiente virtual local (nao versionado)
```

O guia de deploy passo a passo (`DEPLOY_STREAMLIT.md`) fica um nivel
acima desta pasta, em `/Users/guscamar/Desktop/FIAP4/`, junto com os
PDFs do desafio. Ele esta fora de proposito porque o que sera enviado
ao GitHub e ao Streamlit Cloud eh apenas o conteudo desta pasta.

## Como rodar localmente

A venv ja foi criada e as dependencias ja estao instaladas. Para subir o
app basta:

```bash
cd /Users/guscamar/Desktop/FIAP4/desenvolvimento-gustavo
source venv/bin/activate
streamlit run app.py
```

A aplicacao abre em `http://localhost:8501`. No menu lateral aparecem as
duas paginas:

- **Predicao Individual**: formulario que recebe os dados de uma pessoa
  e devolve a classe prevista junto com a confianca por categoria.
- **Dashboard Analitico**: painel com KPIs, distribuicoes, IMC por classe,
  habitos vs obesidade, historico familiar, transporte, correlacao
  numerica e a qualidade do modelo (matriz de confusao).

Se quiser parar o servidor, use `Ctrl + C` no terminal.

### Recriando a venv do zero (caso necessario)

Se a venv quebrar (por exemplo, se a pasta for movida ou copiada para
outra maquina), recrie com:

```bash
cd /Users/guscamar/Desktop/FIAP4/desenvolvimento-gustavo
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Como publicar no Streamlit Community Cloud

Este eh o caminho mais simples para deixar o app online sem custo. O
guia passo a passo com prints fica em `../DEPLOY_STREAMLIT.md` (um
nivel acima desta pasta). Resumo:

1. Suba o conteudo desta pasta para um repositorio publico no GitHub.
2. Acesse `https://share.streamlit.io/` e clique em **New app**.
3. Selecione o repositorio, branch (`main`) e em **Main file path**
   informe `app.py`.
4. Clique em **Deploy**. Em poucos minutos o app estara disponivel em
   uma URL no formato `https://<algumacoisa>.streamlit.app`.

O `requirements.txt` e o `runtime.txt` sao detectados automaticamente, e
o modelo (`best_pipeline.joblib`) ja vai junto, entao nao eh necessario
retreinar no servidor.

## Como subir para o GitHub

Com a pasta ja organizada, basta:

```bash
cd /Users/guscamar/Desktop/FIAP4/desenvolvimento-gustavo

# Inicializa o repositorio local (apenas na primeira vez)
git init -b main

# Configura o autor do commit (opcional, se ainda nao configurou globalmente)
git config user.name "Seu Nome"
git config user.email "seu.email@exemplo.com"

git add .
git commit -m "Versao final do Tech Challenge - Fase 4"

# Cria o repositorio no GitHub (pelo navegador) e cole a URL abaixo
git remote add origin https://github.com/<seu-usuario>/<seu-repo>.git
git push -u origin main
```

A pasta `venv/` esta no `.gitignore`, entao ela nao vai junto. O modelo
serializado (`best_pipeline.joblib`) eh versionado de proposito porque o
Streamlit Cloud precisa dele para servir as predicoes sem retreinar.

## Modelagem e resultados

O script `train_model.py` compara tres algoritmos com `GridSearchCV` e
validacao cruzada estratificada de 5 folds, usando `f1_macro` como
metrica de selecao:

- LogisticRegression (referencia linear)
- RandomForestClassifier (ensemble por bagging)
- GradientBoostingClassifier (ensemble por boosting)

Resultados no holdout (20% da base, 423 registros):

| Modelo               | CV F1-macro | Test Accuracy | Test F1-macro |
|----------------------|-------------|---------------|---------------|
| LogisticRegression   | 0.9420      | 0.9402        | 0.9377        |
| RandomForest         | 0.9762      | 0.9689        | 0.9672        |
| GradientBoosting     | 0.9712      | 0.9713        | 0.9701        |

O **GradientBoosting** foi o vencedor e ficou salvo em
`models/best_pipeline.joblib`. Todos os modelos passaram com folga do
limite minimo de 75% de assertividade exigido pelo desafio.

Para reexecutar o treino do zero:

```bash
source venv/bin/activate
python train_model.py
```

O script imprime o progresso por modelo, regrava o `.joblib` e
atualiza `models/metrics_report.json`.

## Feature engineering

Tres features derivadas alimentam o modelo:

- `imc`: indice de massa corporal classico (`Peso / Altura ** 2`).
- `Healthy_Score`: soma de `FCVC + FAF + CH2O`. Combina alimentacao
  saudavel, atividade fisica e hidratacao em um unico indicador.
- `Sedentary_Index`: razao `TUE / (FAF + 1)`. Indica o quanto o tempo de
  tela domina sobre a atividade fisica. O `+ 1` evita divisao por zero
  quando `FAF` for zero.

Variaveis categoricas com ordem natural de intensidade (`CAEC`, `CALC`,
`MTRANS`) recebem mapeamentos manuais que preservam essa ordem. As
binarias yes/no viram 0/1, e o genero vira `gender_binary` (0 = Male,
1 = Female).

A lista final de features que entram no modelo (na ordem) esta na
constante `FEATURE_COLUMNS` de `train_model.py`.

## Como reabrir e re-executar o notebook

O notebook `notebooks/TechChallenge4_v2.ipynb` ja esta executado e com
saidas salvas. Para reabrir e rodar de novo:

```bash
source venv/bin/activate
pip install jupyter --quiet
jupyter notebook notebooks/TechChallenge4_v2.ipynb
```

Ou, mais pratico, abrir o arquivo direto no VS Code, escolher o kernel
da venv (`./venv/bin/python`) e usar **Run All**. Tudo eh relativo a
posicao do notebook, entao nao tem caminho absoluto que possa quebrar.

## Checklist de entrega

Os entregaveis exigidos pelo Tech Challenge ficam organizados assim:

- [x] Pipeline de Machine Learning completo com feature engineering e
  treinamento do modelo (`train_model.py` + `notebooks/TechChallenge4_v2.ipynb`)
- [x] Modelo com assertividade acima de 75% (atingimos ~97%)
- [x] Deploy do modelo em uma aplicacao Streamlit (`app.py` + `pages/`)
- [x] Painel analitico com insights (pagina **Dashboard Analitico** dentro do app)
- [ ] Repositorio no GitHub com todo o codigo
- [ ] Link da aplicacao publicada no Streamlit
- [ ] Video de apresentacao (4 a 10 minutos)
- [ ] Arquivo `entrega.txt` preenchido com os tres links acima

Os tres ultimos itens dependem das suas acoes externas (subir no GitHub,
publicar no Streamlit, gravar e enviar o video).

## Solucao de problemas

**Erro "Arquivo do modelo nao encontrado" ao abrir a pagina de predicao**

Significa que o arquivo `models/best_pipeline.joblib` nao esta presente.
Rode `python train_model.py` para regerar.

**Streamlit reclama de versao do scikit-learn ao carregar o modelo**

O `.joblib` foi treinado com a versao do `scikit-learn` listada em
`requirements.txt` (`1.5.2`). Se voce mudar a versao, retreine o modelo
com `python train_model.py` antes de subir o app.

**Notebook nao acha o CSV**

O notebook usa `Path('..') / 'data' / 'Obesity.csv'`, ou seja, ele espera
estar dentro de `notebooks/`. Se voce mover o notebook, ajuste o caminho
ou execute a partir da pasta `notebooks/`.

**Quero rodar em outra maquina**

Basta copiar a pasta inteira (sem a `venv/`), recriar a venv com os
comandos da secao [Como rodar localmente](#como-rodar-localmente) e
executar `streamlit run app.py`.
