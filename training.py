"""
training.py

Entrenamiento de XGBoost con búsqueda de hiperparámetros via Optuna.
Registra parámetros, métricas y artefacto del modelo en MLflow.
"""

import json
import os
import pickle
import time
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from config import (
    DROP_COLS, EXPERIMENT_NAME, ID_COLS, MODEL_DIR,
    N_TRIALS, POST_COLS, RANDOM_STATE, TARGET_COL,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DROP_COLS = list(dict.fromkeys(ID_COLS + [TARGET_COL] + POST_COLS + DROP_COLS))


def _split_xy(df: pd.DataFrame):
    """Separa features y target, eliminando columnas de identificación y postproceso."""
    drop = [c for c in _DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET_COL]
    return X, y


def _align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Asegura que train y test tengan exactamente las mismas columnas."""
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0
    return X_train, X_test[X_train.columns]


def _build_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     RANDOM_STATE,
    }


def train_and_log(
    train_path: str,
    test_path: str,
    val_path: str,
    model_dir: str = MODEL_DIR,
    n_trials: int = N_TRIALS,
    experiment_name: str = EXPERIMENT_NAME,
):
    """
    Busca hiperparámetros con Optuna y registra el mejor modelo en MLflow.

    Parameters
    ----------
    train_path : str
        Ruta al CSV de entrenamiento.
    test_path : str
        Ruta al CSV de test (usado en el objetivo de Optuna).
    val_path : str
        Ruta al CSV de validación out-of-time.
    model_dir : str
        Directorio base donde se guardará el artefacto del modelo.
    n_trials : int
        Número de iteraciones de Optuna.
    experiment_name : str
        Nombre del experimento en MLflow.

    Returns
    -------
    run_id : str
    model : XGBClassifier
    """
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df_val   = pd.read_csv(val_path)

    X_train, y_train = _split_xy(df_train)
    X_test,  y_test  = _split_xy(df_test)
    X_val,   y_val   = _split_xy(df_val)

    X_train, X_test = _align_columns(X_train, X_test)
    _,       X_val  = _align_columns(X_train.copy(), X_val)

    def objective(trial: optuna.Trial) -> float:
        params = _build_params(trial)
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    print(f"  Iniciando búsqueda con Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "use_label_encoder": False,
        "eval_metric":       "logloss",
        "random_state":      RANDOM_STATE,
    }

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    auc_test  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
    auc_val   = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])
    decay     = ((auc_train - auc_test) / auc_train) * 100 if auc_train > 0 else float("inf")

    print(f"\n  AUC Train : {auc_train:.4f}")
    print(f"  AUC Test  : {auc_test:.4f}")
    print(f"  AUC Val   : {auc_val:.4f}")
    print(f"  Decay     : {decay:.2f}%")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_test",  auc_test)
        mlflow.log_metric("auc_val",   auc_val)
        mlflow.log_metric("decay_pct", decay)
        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name="cu_venta_xgb",
        )
        run_id = run.info.run_id

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    artifact_dir = os.path.join(model_dir, ts)
    os.makedirs(artifact_dir, exist_ok=True)

    with open(os.path.join(artifact_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "ml_name": "xgb",
        "run_id": run_id,
        "performance": {
            "auc_train":  round(auc_train, 4),
            "auc_test":   round(auc_test, 4),
            "auc_val":    round(auc_val, 4),
            "decay_pct":  round(decay, 4),
        },
        "hyperparameters": best_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(artifact_dir, "xgb_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n  Modelo guardado en: {artifact_dir}")
    return run_id, model