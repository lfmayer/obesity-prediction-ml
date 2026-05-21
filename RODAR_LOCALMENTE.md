# Guia rapido - Rodar o app localmente

Instrucoes para abrir a aplicacao Streamlit no seu computador, a partir
da pasta `desenvolvimento-gustavo/`. Igual ao `DEPLOY_STREAMLIT.md`,
este arquivo fica **fora** da pasta de desenvolvimento de proposito,
como referencia pessoal.

## O que voce precisa

- macOS com `python3` instalado (qualquer versao 3.10 ou superior).
- A pasta `desenvolvimento-gustavo/` no estado atual. Ja contem a venv
  pronta e o modelo treinado.

## Caminho rapido (cenario normal)

A venv ja esta criada e as dependencias instaladas. Para subir o app:

```bash
cd /Users/guscamar/Desktop/FIAP4/desenvolvimento-gustavo
source venv/bin/activate
streamlit run app.py
```

O Streamlit vai imprimir algo como:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Se o navegador nao abrir sozinho, copie o **Local URL** e cole no Chrome
ou Safari.

No menu lateral aparecem duas paginas:

- **Predicao Individual** - formulario de predicao por paciente.
- **Dashboard Analitico** - painel com KPIs, graficos e qualidade do modelo.

Para encerrar o servidor, pressione `Ctrl + C` no terminal.

## Caminho do zero (em outra maquina ou se a venv quebrar)

Use isto se voce moveu a pasta, copiou para outro Mac, ou se o
`source venv/bin/activate` deixar de funcionar:

```bash
cd /Users/guscamar/Desktop/FIAP4/desenvolvimento-gustavo

# Recria a venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Instala as dependencias com versoes fixadas
pip install --upgrade pip
pip install -r requirements.txt

# Sobe o app
streamlit run app.py
```

A primeira instalacao pode levar alguns minutos. Depois disso, basta
ativar a venv e rodar `streamlit run app.py`.

## E se eu nao tiver o modelo treinado?

O arquivo `models/best_pipeline.joblib` ja vem treinado. Se ele tiver
sido apagado por engano, regere com:

```bash
source venv/bin/activate
python train_model.py
```

O treino completo (com `GridSearchCV` nos tres modelos) leva poucos
minutos em um notebook moderno. Ao final, o `.joblib` e o
`metrics_report.json` aparecem em `models/`.

## Como abrir o notebook

O notebook `notebooks/TechChallenge4_v2.ipynb` ja esta executado e com
saidas salvas. Para reabri-lo:

**Opcao A - VS Code (mais simples)**

1. Abra a pasta `desenvolvimento-gustavo/` no VS Code.
2. Abra o arquivo `notebooks/TechChallenge4_v2.ipynb`.
3. No canto superior direito, clique em **Select Kernel** e escolha
   o Python da venv (`./venv/bin/python`).
4. Use **Run All** se quiser reexecutar todas as celulas.

**Opcao B - Jupyter**

```bash
source venv/bin/activate
pip install jupyter --quiet
jupyter notebook notebooks/TechChallenge4_v2.ipynb
```

## Dicas e atalhos

- Para sair da venv, basta digitar `deactivate` no terminal.
- Para ver as portas em uso (caso `8501` esteja ocupada):
  `lsof -nP -iTCP:8501 | grep LISTEN`. Voce pode tambem subir o app
  em outra porta com `streamlit run app.py --server.port 8765`.
- O Streamlit recarrega o app automaticamente quando voce salva uma
  alteracao em `app.py` ou nos arquivos de `pages/`. Basta atualizar
  a aba do navegador.

## Problemas comuns

**`command not found: streamlit`**

Voce esqueceu de ativar a venv. Rode `source venv/bin/activate` antes.

**`zsh: command not found: source` ou caminho invalido**

Voce nao esta dentro de `desenvolvimento-gustavo/`. Confira com `pwd`
e use `cd /Users/guscamar/Desktop/FIAP4/desenvolvimento-gustavo`.

**`ModuleNotFoundError: No module named 'streamlit'`**

A venv ativa nao tem as libs. Rode:

```bash
pip install -r requirements.txt
```

**Erro `Arquivo do modelo nao encontrado` na pagina de predicao**

O `.joblib` foi apagado. Regere com `python train_model.py`.

**Porta 8501 ja em uso**

Outro processo do Streamlit esta rodando. Encerre com `Ctrl + C` no
terminal anterior, ou use outra porta:

```bash
streamlit run app.py --server.port 8765
```
