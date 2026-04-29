"""
preprocessing.py

Descarga (opcional), carga y limpieza del dataset raw.
Genera los splits de entrenamiento, test y validación temporal.
"""

import glob
import os

import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CODMES_COL, DROP_COLS, NAN_THRESHOLD, POST_COLS, RANDOM_STATE,
    TARGET_COL, TEST_SIZE, VALIDATION_MONTH,
    VARS_CATEGORICAS, VARS_NUMERICAS,
)


def download_from_drive(folder_url: str, dest_dir: str) -> None:
    """
    Descarga todos los archivos de una carpeta pública de Google Drive.

    Parameters
    ----------
    folder_url : str
        URL de la carpeta de Drive.
    dest_dir : str
        Directorio local donde se guardarán los archivos.
    """
    os.makedirs(dest_dir, exist_ok=True)
    gdown.download_folder(folder_url, output=dest_dir, quiet=False, use_cookies=False)


def load_raw_fragments(raw_dir: str) -> pd.DataFrame:
    """
    Lee y concatena todos los fragmentos CSV encontrados en raw_dir.

    Parameters
    ----------
    raw_dir : str
        Directorio que contiene los archivos CSV.

    Returns
    -------
    pd.DataFrame
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {raw_dir}")

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            print(f"  No se pudo leer {f}: {exc}")

    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(frames)} fragmentos cargados — {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def drop_high_nan(df: pd.DataFrame, threshold: float = NAN_THRESHOLD):
    """
    Elimina columnas cuyo porcentaje de valores nulos supera el umbral.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Porcentaje máximo de NaN permitido (0–100).

    Returns
    -------
    df_clean : pd.DataFrame
    dropped : list[str]
    """
    dropped = [c for c in df.columns if df[c].isna().mean() * 100 > threshold]
    df = df.drop(columns=dropped)
    if dropped:
        print(f"  {len(dropped)} columnas eliminadas (>{threshold}% NaN): {dropped}")
    return df, dropped


def impute_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa con -9999999 las variables numéricas del modelo y las castea a float32.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.replace(["", "null", "None"], np.nan)

    present = [v for v in VARS_NUMERICAS if v in df.columns]
    df[present] = df[present].fillna(-9999999).astype("float32")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa y one-hot-encodea las variables categóricas con categorías fijas.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    for col, valid_vals in VARS_CATEGORICAS.items():
        if col not in df.columns:
            continue

        df[col] = df[col].astype("string").fillna("SV")
        df.loc[~df[col].isin(valid_vals), col] = "OTRO"

        categories = valid_vals + ["OTRO"]
        dummies = pd.get_dummies(
            df[col].astype(pd.CategoricalDtype(categories)),
            prefix=col,
        )
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


def run_preprocessing(
    raw_dir: str,
    processed_dir: str,
    drive_url: str | None = None,
    nan_threshold: float = NAN_THRESHOLD,
    validation_month: float = VALIDATION_MONTH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Parameters
    ----------
    raw_dir : str
        Carpeta con los CSVs crudos.
    processed_dir : str
        Carpeta de destino para los splits procesados.
    drive_url : str, optional
        URL de Drive para descarga previa del raw data.
    nan_threshold : float
        Umbral de NaN (%) para eliminar columnas.
    validation_month : int
        Valor de p_fecinformacion reservado para validación out-of-time (formato YYYYMMDD).
    test_size : float
        Proporción del conjunto de entrenamiento usada para test.
    random_state : int
        Semilla de reproducibilidad.

    Returns
    -------
    df_train, df_test, df_val : pd.DataFrame
    metadata : dict
    """
    if drive_url:
        print("  Descargando datos desde Google Drive...")
        download_from_drive(drive_url, raw_dir)

    df = load_raw_fragments(raw_dir)

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Columnas no utilizables eliminadas: {cols_to_drop}")

    df, dropped = drop_high_nan(df, nan_threshold)
    df = impute_numerics(df)
    df = encode_categoricals(df)

    if CODMES_COL not in df.columns:
        raise KeyError(
            f"Columna '{CODMES_COL}' no encontrada. "
            "Verificar el nombre del campo de período en el raw data."
        )

    df_val  = df[df[CODMES_COL] == validation_month].copy()
    df_main = df[df[CODMES_COL] != validation_month].copy()

    stratify_col = df_main[TARGET_COL] if TARGET_COL in df_main.columns else None
    df_train, df_test = train_test_split(
        df_main,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    os.makedirs(processed_dir, exist_ok=True)
    df_train.to_csv(os.path.join(processed_dir, "df_train.csv"), index=False)
    df_test.to_csv(os.path.join(processed_dir, "df_test.csv"),  index=False)
    df_val.to_csv(os.path.join(processed_dir, "df_val.csv"),   index=False)

    print(f"  Train : {df_train.shape[0]:,} filas")
    print(f"  Test  : {df_test.shape[0]:,} filas")
    print(f"  Val   : {df_val.shape[0]:,} filas")

    metadata = {
        "dropped_cols":      dropped,
        "validation_month":  validation_month,
        "n_train":           len(df_train),
        "n_test":            len(df_test),
        "n_val":             len(df_val),
    }
    return df_train, df_test, df_val, metadata
