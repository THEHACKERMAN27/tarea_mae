"""
postprocessing.py

Cálculo de puntuación TLV, segmentación en grupos de ejecución
y generación del archivo de réplica pipe-delimitado.

Fórmula TLV:
    puntuacion_tlv = prob × prob_value_contact × log(monto + 1) × prob_frescura
"""

import datetime
import os

import numpy as np
import pandas as pd

from config import DIST_GE, POST_DIR, REPLICA_DIR


def get_groups(scores: np.ndarray, df_post: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la puntuación TLV y asigna el grupo de ejecución (1–10).

    Parameters
    ----------
    scores : array-like
        Probabilidades predichas por el modelo [0, 1].
    df_post : pd.DataFrame
        DataFrame con columnas: grp_campecs06m, prob_value_contact, monto.

    Returns
    -------
    pd.DataFrame con columnas adicionales:
        prob, prob_frescura, puntuacion_tlv, grupo_ejec_tlv.
    """
    df_post["prob"] = scores

    df_post["prob_frescura"] = np.where(df_post["grp_campecs06m"] == "G1", 0.066,
                               np.where(df_post["grp_campecs06m"] == "G2", 0.028,
                               np.where(df_post["grp_campecs06m"] == "G3", 0.022,
                               np.where(df_post["grp_campecs06m"] == "G4", 0.008, 0.004))))

    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)

    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )

    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"],
        q=DIST_GE,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )

    return df_post


def run_postprocessing(
    scores: np.ndarray,
    df_post: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Wrapper de get_groups con exportación opcional a CSV.

    Parameters
    ----------
    scores : array-like
        Probabilidades del modelo.
    df_post : pd.DataFrame
        DataFrame con variables de postproceso.
    output_path : str, optional
        Ruta donde guardar el resultado.

    Returns
    -------
    pd.DataFrame
    """
    result = get_groups(scores, df_post.copy())

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"  Output TLV guardado en: {output_path}")

    return result


def save_replica(
    df_post: pd.DataFrame,
    table: str,
    partition: str,
    dir_s3: str        = os.path.join(REPLICA_DIR, "s3"),
    dir_athena: str    = os.path.join(REPLICA_DIR, "athena"),
    dir_onpremise: str = os.path.join(REPLICA_DIR, "onpremise"),
) -> None:
    """
    Genera el archivo de réplica pipe-delimitado para tres destinos.

    Columnas de salida:
        codmes | tipdoc | coddoc | puntuacion | modelo |
        fec_replica | grupo_ejec | score | orden |
        variable1 | variable2 | variable3

    Parameters
    ----------
    df_post : pd.DataFrame
        DataFrame con scores TLV y grupos asignados.
    table : str
        Identificador del modelo para el nombre del archivo.
    partition : str
        Período de la partición (ej. '201912').
    dir_s3, dir_athena, dir_onpremise : str
        Directorios de destino para cada canal.
    """
    codmes_col = "partition" if "partition" in df_post.columns else "p_fecinformacion"
    tipdoc_val = df_post["tip_doc"].astype(str) if "tip_doc" in df_post.columns else "1"

    df_replica = pd.DataFrame({
        "codmes":      df_post[codmes_col],
        "tipdoc":      tipdoc_val,
        "coddoc":      df_post["key_value"],
        "puntuacion":  df_post["puntuacion_tlv"],
        "modelo":      "EC OMNICANAL",
        "fec_replica": datetime.date.today().strftime("%Y%m%d"),
        "grupo_ejec":  df_post["grupo_ejec_tlv"],
        "score":       df_post["prob"],
        "orden":       "",
        "variable1":   df_post["codunicocli"].apply(lambda x: str(x).zfill(10)),
        "variable2":   df_post["monto"],
        "variable3":   "",
    })

    df_replica = (
        df_replica
        .sort_values("puntuacion", ascending=False)
        .drop_duplicates("coddoc", keep="first")
    )
    df_replica["orden"] = (
        df_replica["puntuacion"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    for dest in [dir_s3, dir_athena, dir_onpremise]:
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, f"scr_{table}_{partition}.txt")
        df_replica.to_csv(path, index=False, sep="|")
        print(f"  Réplica guardada en: {path}")
