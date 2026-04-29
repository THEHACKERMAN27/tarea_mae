DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1a83rc3ZuuEn-kVqlizoRiT9hJM2M3Hbw"

RAW_DIR         = "data/raw"
PROCESSED_DIR   = "data/processed"
POST_DIR        = "data/postprocessed"
REPLICA_DIR     = "data/replica"
MONITORING_DIR  = "data/monitoring"
MODEL_DIR       = "data/models"

CODMES_COL       = "p_fecinformacion"   # columna de período en el raw data (formato YYYYMMDD)
VALIDATION_MONTH = 20220301             # período reservado para validación out-of-time
TARGET_COL       = "target"
NAN_THRESHOLD    = 80
TEST_SIZE        = 0.30
RANDOM_STATE     = 123

ID_COLS   = ["p_fecinformacion", "key_value"]
POST_COLS = ["partition", "tip_doc", "key_value", "codunicocli",
             "grp_campecs06m", "prob_value_contact", "monto"]
DROP_COLS = ["fch_creacion"]            # columnas no utilizables como features (timestamps, etc.)

VARS_NUMERICAS = [
    "nro_producto_6m", "prom_uso_tc_rccsf3m", "ctd_sms_received",
    "max_usotcribksf06m", "ctd_camptot06m", "dsv_svppallsf06m",
    "prm_svprmecs06m", "ctd_app_productos_m1", "ctd_campecsm01",
    "lin_tcrrstsf03m", "mnt_ptm", "dif_no_gestionado_4meses",
    "max_campecs06m", "beta_pctusotcr12m", "rat_disefepnm01",
    "flg_saltotppe12m", "prom_sow_lintcribksf3m", "openhtml_1m",
    "nprod_1m", "nro_transfer_6m", "max_usotcrrstsf03m",
    "prm_cnt_fee_amt_u7d", "pas_avg6m_max12m", "beta_saltotppe12m",
    "seg_un", "ant_ultprdallsf", "avg_sald_pas_3m", "pas_1m_avg3m",
    "num_incrsaldispefe06m", "cnl_age_p4m_p12m", "cnl_atm_p4m_p12m",
    "cre_lin_tc_rccibk_m07", "prm_svprmlibdis06m", "ingreso_neto",
    "max_nact_12m", "cre_sldtotfinprm03", "dif_contacto_efectivo_10meses",
    "act_1m_avg3m", "monto_consumos_ecommerce_tc", "ctd_camptotm01",
    "prop_atm_4m", "prom_pct_saldopprcc6m", "apppag_1m",
    "nro_configuracion_6m", "act_avg6m_max12m", "sldvig_tcrsrcf",
    "prom_score_acepta_12meses", "telefonos_6meses", "pas_1m_avg6m",
    "ctd_camptototrcnl06m", "prm_saltotrdpj03m", "bpitrx_1m",
    "prm_lintcribksf03m", "ctd_entrdm01", "avg_openhtml_6m",
    "tea", "pct_usotcrm01", "senthtml_1m",
]

VARS_CATEGORICAS = {
    "ent_1erlntcrallsfm01": ["INTERBANK"],
}

DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]

N_TRIALS        = 3
EXPERIMENT_NAME = "cu_venta_e2e"
