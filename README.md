# Pipeline ML E2E — CU Venta

Pipeline de machine learning end-to-end para el modelo de propensión **EC OMNICANAL**.  
Cubre ingesta, preprocesamiento, entrenamiento con búsqueda de hiperparámetros, monitoreo de deriva, postprocesamiento con scoring TLV y despliegue mediante API REST.

---

## Estructura del proyecto

```
ml-pipeline/
├── main.py              ← Orquestador principal del pipeline
├── config.py            ← Constantes y rutas centralizadas
├── preprocessing.py     ← Limpieza, imputación, encoding y splits
├── training.py          ← XGBoost + Optuna + MLflow
├── monitoring.py        ← PSI, AUC y Recall por decil
├── postprocessing.py    ← Scoring TLV, grupos de ejecución y réplica
├── api.py               ← API REST de inferencia (FastAPI)
├── dashboard.py         ← Dashboard interactivo (Streamlit)
├── Dockerfile           ← Contenedor para la API de inferencia
├── requirements.txt
├── .gitignore
└── data/
    ├── raw/             ← Fragmentos CSV del dataset crudo
    ├── processed/       ← df_train.csv, df_test.csv, df_val.csv
    ├── postprocessed/   ← output_tlv.csv
    ├── replica/         ← Archivos pipe-delimitados por canal
    │   ├── s3/
    │   ├── athena/
    │   └── onpremise/
    ├── monitoring/      ← recall_by_decile.csv
    └── models/          ← Artefactos del modelo por timestamp
```

---

## 🛠️ Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/THEHACKERMAN27/tarea_mae.git
   cd tarea_mae
   ```

2. **Crear un entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución

### 1 — Pipeline completo

**Con descarga automática desde Google Drive:**

```bash
python main.py --download
```

**Con los CSV ya descargados en `data/raw/`:**

```bash
python main.py
```

El pipeline ejecuta las cuatro etapas en orden: preprocesamiento → entrenamiento → monitoreo → postprocesamiento.

### 2 — API de inferencia (local)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva (Swagger UI): `http://localhost:8000/docs`

### 3 — API de inferencia (Docker)

```bash
docker build -t cu-venta-api .
docker run -p 8000:8000 cu-venta-api
```

### 4 — Dashboard

```bash
streamlit run dashboard.py
```

---

## Configuración

Todos los parámetros se centralizan en `config.py`. Los más relevantes:

| Parámetro          | Descripción                                              | Valor actual       |
|--------------------|----------------------------------------------------------|--------------------|
| `CODMES_COL`       | Columna de período en el raw data                        | `p_fecinformacion` |
| `VALIDATION_MONTH` | Período reservado para validación out-of-time (YYYYMMDD) | `20220301`         |
| `NAN_THRESHOLD`    | % de NaN máximo para conservar una columna               | `80`               |
| `TEST_SIZE`        | Proporción del split de test                             | `0.30`             |
| `N_TRIALS`         | Iteraciones de búsqueda de Optuna                        | `30`               |
| `EXPERIMENT_NAME`  | Nombre del experimento en MLflow                         | `cu_venta_e2e`     |

> **Importante:** antes de ejecutar, verifica que `VALIDATION_MONTH` corresponde al período
> más reciente de tu dataset. Puedes revisarlo con:
> ```python
> import pandas as pd, glob
> df = pd.concat([pd.read_csv(f) for f in glob.glob("data/raw/*.csv")])
> print(sorted(df["p_fecinformacion"].unique()))
> ```
> Luego actualiza el valor en `config.py` si es necesario.

---

## Estructura del dataset

| Columna | Tipo | Rol |
|---|---|---|
| `partition` | string | Identificador de partición (`p1`, `p2`, …) |
| `tip_doc` | int | Tipo de documento |
| `key_value` | string (hash) | Identificador único del cliente |
| `codunicocli` | int | Código único del cliente |
| `p_fecinformacion` | int (YYYYMMDD) | Período — usado para el split temporal |
| `fch_creacion` | timestamp | Fecha de creación — eliminada antes del entrenamiento |
| `target` | int (0/1) | Variable objetivo |
| `grp_campecs06m` | string | Grupo de campaña — usado en scoring TLV |
| `prob_value_contact` | float | Probabilidad de contacto efectivo — usado en scoring TLV |
| `monto` | int | Monto — usado en scoring TLV |
| `ent_1erlntcrallsfm01` | string | Variable categórica del modelo (one-hot encoding) |
| resto (58 columnas) | float | Variables numéricas del modelo |

Las columnas `partition`, `tip_doc`, `key_value`, `codunicocli`, `grp_campecs06m`, `prob_value_contact` y `monto` se conservan en los splits para el postprocesamiento, pero no se pasan al modelo como features.

---

## Módulos

### `config.py`
Punto único de verdad para rutas, columnas, listas de variables y constantes de entrenamiento. Cualquier ajuste al pipeline empieza aquí.

### `preprocessing.py`
- Descarga opcional desde Google Drive (`gdown`).
- Unifica múltiples fragmentos CSV en un único DataFrame.
- Elimina `fch_creacion` y otras columnas no utilizables definidas en `DROP_COLS`.
- Elimina columnas con más del 80% de valores nulos.
- Imputa variables numéricas con centinela `-9999999`.
- One-hot encoding de `ent_1erlntcrallsfm01` con categorías fijas (`INTERBANK` / `OTRO`).
- Split temporal: validación fija en `p_fecinformacion == 20220301`; train/test estratificado por target sobre el resto.

### `training.py`
- Búsqueda de hiperparámetros con **Optuna** (optimización bayesiana, 30 trials).
- Objetivo: maximizar AUC en el conjunto de test.
- Re-entrenamiento final con los mejores parámetros encontrados.
- Reporta AUC en train, test y validación out-of-time, más el porcentaje de decay.
- Registra experimento en **MLflow** (parámetros, métricas, artefacto del modelo).
- Guarda modelo `.pkl` + metadata `.json` en `data/models/<timestamp>/`.

### `monitoring.py`
- **PSI** (Population Stability Index) entre distribución de scores de train y validación.
  - `< 0.10` → OK &nbsp;·&nbsp; `0.10–0.25` → WARN &nbsp;·&nbsp; `> 0.25` → ALERT
- **AUC** sobre el conjunto out-of-time.
- **Recall acumulado por decil** (decil 1 = clientes con score más alto).

### `postprocessing.py`
- Calcula la puntuación TLV compuesta:
  ```
  puntuacion_tlv = prob × prob_value_contact × log(monto + 1) × prob_frescura
  ```
- Asigna `grupo_ejec_tlv` (1–10) usando cuantiles de negocio fijos (`DIST_GE`).
- Genera el archivo de réplica pipe-delimitado (`|`) para tres destinos: S3, Athena y on-premise.

### `api.py`
API REST construida con **FastAPI**. Carga el modelo más reciente al iniciar el servicio.

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado del servicio y modelo cargado |
| `GET` | `/model/info` | Metadata completa del modelo activo |
| `POST` | `/predict` | Predicción individual (un cliente) |
| `POST` | `/predict/batch` | Predicción en lote (hasta 10 000 registros) |

Los campos no enviados en `/predict` se imputan automáticamente con `-9999999`.

### `dashboard.py`
Visualización interactiva con **Streamlit**: KPIs del modelo, distribución de grupos de ejecución, recall por decil, top clientes por puntuación TLV y metadata del modelo entrenado.

---

## Dataset

- **Fuente:** [Google Drive](https://drive.google.com/drive/folders/1a83rc3ZuuEn-kVqlizoRiT9hJM2M3Hbw)
- **Formato:** múltiples fragmentos CSV, un archivo por partición (`p1`, `p2`, …)
- **Target:** columna binaria `target`
- **Período de validación:** `p_fecinformacion == 20220301`
- **Features del modelo:** 58 numéricas + 1 categórica (2 dummies tras encoding) = 60 features

---

## MLflow

El servidor de MLflow se levanta automáticamente sobre la carpeta `mlruns/` al ejecutar el pipeline. Para visualizar los experimentos:

```bash
mlflow ui
```

Acceder en: `http://localhost:5000`