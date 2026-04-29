"""
monitoring.py

Monitoreo de estabilidad poblacional (PSI), AUC y Recall por decil.

Umbrales PSI:
    < 0.10          -> OK   (sin deriva)
    0.10  – 0.25    -> WARN (deriva moderada)
    > 0.25          -> ALERT (deriva severa)
"""

import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

from config import MONITORING_DIR, TARGET_COL


def psi_flag(psi: float) -> str:
    """Clasifica el nivel de alerta según el valor de PSI."""
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARN"
    return "ALERT"


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones de scores.

    Parameters
    ----------
    expected : array-like
        Distribución de referencia (scores de entrenamiento).
    actual : array-like
        Distribución actual (scores de validación).
    n_bins : int
        Número de bins cuantílicos.

    Returns
    -------
    float
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    breakpoints        = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints[0]     = -np.inf
    breakpoints[-1]    = np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = np.where(exp_counts == 0, 1e-6, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / len(actual))

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 6)


def compute_recall_by_decile(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Calcula el Recall acumulado por decil de score (decil 1 = scores más altos).

    Parameters
    ----------
    y_true : array-like
        Valores reales del target binario.
    scores : array-like
        Probabilidades predichas por el modelo.
    n_deciles : int
        Número de grupos cuantílicos.

    Returns
    -------
    pd.DataFrame con columnas: decil, n_clientes, positivos, recall_acumulado
    """
    df = (
        pd.DataFrame({"score": scores, "target": y_true})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    df["decil"] = pd.qcut(
        df.index,
        q=n_deciles,
        labels=range(1, n_deciles + 1),
    ).astype(int)

    total_positivos = df["target"].sum()
    records = []
    for d in range(1, n_deciles + 1):
        subset = df[df["decil"] <= d]
        positivos_acum = subset["target"].sum()
        recall = positivos_acum / total_positivos if total_positivos > 0 else 0.0
        records.append({
            "decil":            d,
            "n_clientes":       len(subset),
            "positivos_acum":   int(positivos_acum),
            "recall_acumulado": round(recall, 4),
        })

    return pd.DataFrame(records)


def run_monitoring(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    val_scores: np.ndarray,
    train_scores: np.ndarray | None = None,
    target_col: str = TARGET_COL,
    output_dir: str = MONITORING_DIR,
    mlflow_active: bool = False,
) -> dict:
    """
    Ejecuta PSI, AUC y Recall por decil sobre el conjunto de validación.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset de entrenamiento (distribución de referencia).
    df_val : pd.DataFrame
        Dataset de validación.
    val_scores : array-like
        Scores predichos sobre el conjunto de validación.
    train_scores : array-like, optional
        Scores predichos sobre entrenamiento (requerido para PSI).
    target_col : str
        Nombre de la columna target.
    output_dir : str
        Directorio donde se guardarán los reportes.
    mlflow_active : bool
        Si True, loguea métricas en el run activo de MLflow.

    Returns
    -------
    dict con claves: auc_val, psi_score, flag
    """
    os.makedirs(output_dir, exist_ok=True)

    auc_val    = roc_auc_score(df_val[target_col], val_scores)
    recall_df  = compute_recall_by_decile(df_val[target_col].values, val_scores)
    results    = {"auc_val": round(auc_val, 4), "psi_score": None, "flag": None}

    print(f"  AUC Validación : {auc_val:.4f}")

    if train_scores is not None:
        psi  = compute_psi(train_scores, val_scores)
        flag = psi_flag(psi)
        results["psi_score"] = psi
        results["flag"]      = flag
        print(f"  PSI            : {psi:.4f}  [{flag}]")

    recall_path = os.path.join(output_dir, "recall_by_decile.csv")
    recall_df.to_csv(recall_path, index=False)
    print(f"\n  Recall por decil:\n{recall_df.to_string(index=False)}")

    if mlflow_active:
        mlflow.log_metric("auc_val", auc_val)
        if results["psi_score"] is not None:
            mlflow.log_metric("psi_score", results["psi_score"])

    return results
