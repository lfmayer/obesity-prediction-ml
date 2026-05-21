# =============================================================================
# train_model.py
# -----------------------------------------------------------------------------
# Script responsavel por:
#   1. Carregar a base de dados Obesity.csv.
#   2. Aplicar as etapas de limpeza e feature engineering.
#   3. Comparar tres modelos (Logistic Regression, Random Forest, Gradient
#      Boosting) usando GridSearchCV com validacao cruzada estratificada.
#   4. Selecionar o melhor pipeline e salva-lo em models/best_pipeline.joblib.
#   5. Gerar um relatorio de avaliacao em models/metrics_report.json.
#
# Esse script eh chamado tanto pelo notebook (para reproducao) quanto pelo
# dashboard analitico (que usa as mesmas funcoes de preparacao). A logica eh
# centralizada para evitar divergencia entre o que vai pro modelo e o que
# aparece nas analises.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Caminhos do projeto
# -----------------------------------------------------------------------------
# Tudo eh relativo a localizacao deste arquivo para que o script funcione
# independente do diretorio onde for executado (local, Colab ou Streamlit).
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Obesity.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics_report.json"

# -----------------------------------------------------------------------------
# Constantes globais usadas em varias partes do projeto
# -----------------------------------------------------------------------------

# Lista das classes do alvo, na ordem alfabetica que o sklearn usa internamente.
# Manter explicito ajuda quando precisamos casar resultado/probabilidades com
# os nomes apresentados ao usuario final.
TARGET_CLASSES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II",
]

# Colunas numericas que, segundo o dicionario de dados, sao categoricas
# discretas mas vem do CSV com ruido (decimais). Sao arredondadas para int.
COLS_TO_ROUND_INT = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Mapeamentos manuais (preservam a ordem de "intensidade" da variavel)
MTRANS_MAP = {
    "Walking": 3,
    "Bike": 3,
    "Public_Transportation": 2,
    "Automobile": 1,
    "Motorbike": 1,
}
CAEC_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
GENDER_MAP = {"Male": 0, "Female": 1}
YESNO_MAP = {"no": 0, "yes": 1}

# Colunas binarias yes/no que viraram 0/1
BINARY_COLUMNS = ["family_history", "FAVC", "SMOKE", "SCC"]

# Lista final de features que entram no modelo (ordem importa, eh ela que
# vai bater com o feature_names_in_ do pipeline na hora de servir predicoes).
FEATURE_COLUMNS = [
    "Age",
    "Height",
    "NCP",
    "imc",
    "Healthy_Score",
    "Sedentary_Index",
    "MTRANS_Code",
    "caec_code",
    "calc_code",
    "gender_binary",
    "family_history",
    "FAVC",
    "SMOKE",
    "SCC",
]

# Colunas numericas que serao padronizadas pelo StandardScaler. As demais
# colunas ja estao em escalas pequenas (0-3) e ficam como passthrough.
NUMERIC_COLUMNS = ["Age", "Height", "NCP", "imc", "Healthy_Score", "Sedentary_Index"]


# -----------------------------------------------------------------------------
# Funcoes de preparacao da base
# -----------------------------------------------------------------------------
def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega o CSV bruto e remove duplicatas exatas se existirem."""
    df = pd.read_csv(path)
    # Garante que nao ha duplicatas que possam vazar entre treino e teste.
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def clean_decimal_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Arredonda para inteiro as colunas que sao categoricas mas vem com
    decimais (Age, FCVC, NCP, CH2O, FAF, TUE), conforme orienta o dicionario
    de dados."""
    df = df.copy()
    df[COLS_TO_ROUND_INT] = df[COLS_TO_ROUND_INT].round().astype(int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as features derivadas usadas pelo modelo:
        - imc: indice de massa corporal classico (kg/m**2)
        - Healthy_Score: combinacao simples de habitos saudaveis
        - Sedentary_Index: razao entre tempo de tela e atividade fisica
    """
    df = df.copy()
    df["imc"] = df["Weight"] / (df["Height"] ** 2)
    df["Healthy_Score"] = df["FCVC"] + df["FAF"] + df["CH2O"]
    # Soma 1 no denominador para nao dividir por zero quando FAF == 0.
    df["Sedentary_Index"] = df["TUE"] / (df["FAF"] + 1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos os encodings categoricos do projeto. Mantemos um esquema
    manual (em vez de OneHotEncoder cego) porque varias variaveis tem ordem
    natural de intensidade (CAEC, CALC, MTRANS) e queremos preservar isso."""
    df = df.copy()
    df["MTRANS_Code"] = df["MTRANS"].map(MTRANS_MAP)
    df["caec_code"] = df["CAEC"].map(CAEC_MAP)
    df["calc_code"] = df["CALC"].map(CALC_MAP)
    df["gender_binary"] = df["Gender"].map(GENDER_MAP)
    for col in BINARY_COLUMNS:
        df[col] = df[col].map(YESNO_MAP)
    return df


def build_feature_frame(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pipeline completo de preparacao em uma unica chamada. Retorna X e y
    prontos para o GridSearchCV. Essa funcao eh o ponto de entrada usado
    pelo notebook e pelo dashboard."""
    df = clean_decimal_categoricals(df_raw)
    df = add_engineered_features(df)
    df = encode_categoricals(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df["Obesity"].copy()
    return X, y


# -----------------------------------------------------------------------------
# Construcao do pipeline e dos modelos candidatos
# -----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Cria o pre-processador que aplica StandardScaler nas colunas continuas
    e deixa as demais colunas inalteradas."""
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), NUMERIC_COLUMNS)],
        remainder="passthrough",
    )


def get_candidate_models() -> dict:
    """Define os tres modelos candidatos com suas grades de hiperparametros.
    A escolha foi pensada para ser representativa: um modelo linear, um
    bagging (RandomForest) e um boosting (GradientBoosting)."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [150, 300],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        },
    }


# -----------------------------------------------------------------------------
# Funcao principal de treino
# -----------------------------------------------------------------------------
def train_and_select_best(verbose: bool = True) -> dict:
    """Roda toda a comparacao entre os tres modelos, escolhe o melhor pelo
    f1_macro de validacao cruzada, refita no conjunto completo de treino e
    salva tudo em disco. Retorna um dicionario com o resumo das metricas."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Carrega e prepara os dados -------------------------------------------
    df_raw = load_raw_data()
    X, y = build_feature_frame(df_raw)

    # Split estratificado para preservar a proporcao das classes.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Validacao cruzada estratificada com 5 folds.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = get_candidate_models()
    results = {}
    best_overall = {"name": None, "score": -np.inf, "estimator": None}

    # -- Itera sobre os modelos candidatos ------------------------------------
    for name, cfg in candidates.items():
        if verbose:
            print(f"\n[treino] Avaliando modelo: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", cfg["estimator"]),
            ]
        )

        # GridSearch usa f1_macro porque o dataset tem classes razoavelmente
        # equilibradas mas nao queremos privilegiar a maioria.
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        # Avaliacao no conjunto de teste (holdout).
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results[name] = {
            "best_params": grid.best_params_,
            "cv_f1_macro": float(grid.best_score_),
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=TARGET_CLASSES
            ).tolist(),
        }

        if verbose:
            print(
                f"  - melhores params: {grid.best_params_}\n"
                f"  - cv f1_macro:    {grid.best_score_:.4f}\n"
                f"  - test accuracy:  {acc:.4f}\n"
                f"  - test f1_macro:  {f1:.4f}"
            )

        if f1 > best_overall["score"]:
            best_overall = {"name": name, "score": f1, "estimator": grid.best_estimator_}

    # -- Persiste o melhor modelo --------------------------------------------
    joblib.dump(best_overall["estimator"], MODEL_PATH)

    summary = {
        "best_model": best_overall["name"],
        "best_test_f1_macro": float(best_overall["score"]),
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": TARGET_CLASSES,
        "results_by_model": results,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n[treino] Melhor modelo: {best_overall['name']}")
        print(f"[treino] Modelo salvo em: {MODEL_PATH}")
        print(f"[treino] Relatorio salvo em: {METRICS_PATH}")

    return summary


# -----------------------------------------------------------------------------
# Permite executar o script diretamente: `python train_model.py`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_select_best(verbose=True)
