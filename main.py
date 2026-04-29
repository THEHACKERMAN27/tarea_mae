"""
main.py

Orquestador del pipeline ML E2E: preprocesamiento, entrenamiento,
monitoreo, postprocesamiento y generación de réplica.

Uso:
    python main.py
    python main.py --download       # descarga el raw data desde Drive primero
"""

import argparse
import os

from config import (
    DRIVE_FOLDER_URL, DROP_COLS, ID_COLS, MODEL_DIR, MONITORING_DIR,
    POST_COLS, POST_DIR, PROCESSED_DIR, RAW_DIR, REPLICA_DIR, TARGET_COL,
)
from monitoring import run_monitoring
from postprocessing import run_postprocessing, save_replica
from preprocessing import run_preprocessing
from training import train_and_log

_DROP_FOR_PREDICT = list(dict.fromkeys(ID_COLS + [TARGET_COL] + POST_COLS + DROP_COLS))


def _feature_matrix(df):
    """Devuelve solo las columnas de entrada al modelo."""
    drop = [c for c in _DROP_FOR_PREDICT if c in df.columns]
    return df.drop(columns=drop)


def main(download: bool = False) -> None:
    sep = "=" * 60

    print(sep)
    print("PIPELINE ML E2E — CU VENTA")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Preprocesamiento
    # ------------------------------------------------------------------
    print("\n[1/4] Preprocesamiento")
    df_train, df_test, df_val, prep_meta = run_preprocessing(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        drive_url=DRIVE_FOLDER_URL if download else None,
    )

    # ------------------------------------------------------------------
    # 2. Entrenamiento con Optuna
    # ------------------------------------------------------------------
    print("\n[2/4] Entrenamiento")
    run_id, model = train_and_log(
        train_path=os.path.join(PROCESSED_DIR, "df_train.csv"),
        test_path=os.path.join(PROCESSED_DIR, "df_test.csv"),
        val_path=os.path.join(PROCESSED_DIR, "df_val.csv"),
        model_dir=MODEL_DIR,
    )

    # ------------------------------------------------------------------
    # 3. Monitoreo
    # ------------------------------------------------------------------
    print("\n[3/4] Monitoreo")
    X_train = _feature_matrix(df_train)
    X_val   = _feature_matrix(df_val)

    # Alinear columnas por si hay diferencias tras el split
    for col in set(X_train.columns) - set(X_val.columns):
        X_val[col] = 0
    X_val = X_val[X_train.columns]

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores   = model.predict_proba(X_val)[:, 1]

    monitoring_results = run_monitoring(
        df_train=df_train,
        df_val=df_val,
        val_scores=val_scores,
        train_scores=train_scores,
        output_dir=MONITORING_DIR,
    )

    # ------------------------------------------------------------------
    # 4. Postprocesamiento y réplica
    # ------------------------------------------------------------------
    print("\n[4/4] Postprocesamiento")
    post_output = os.path.join(POST_DIR, "output_tlv.csv")
    df_resultado = run_postprocessing(val_scores, df_val, post_output)

    partition_label = str(df_val["partition"].iloc[0]) if "partition" in df_val.columns \
                      else str(df_val["p_fecinformacion"].iloc[0])

    save_replica(
        df_post=df_resultado,
        table="EC_OMNICANAL",
        partition=partition_label,
        dir_s3=os.path.join(REPLICA_DIR, "s3"),
        dir_athena=os.path.join(REPLICA_DIR, "athena"),
        dir_onpremise=os.path.join(REPLICA_DIR, "onpremise"),
    )

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Pipeline finalizado.")
    print(f"  Run ID MLflow : {run_id}")
    print(f"  AUC Val       : {monitoring_results['auc_val']}")
    if monitoring_results["psi_score"] is not None:
        print(f"  PSI           : {monitoring_results['psi_score']}  [{monitoring_results['flag']}]")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML E2E — CU Venta")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Descarga el raw data desde Google Drive antes de ejecutar.",
    )
    args = parser.parse_args()
    main(download=args.download)