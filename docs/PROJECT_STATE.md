# PROJECT_STATE — model (clasificador de noticias en español)

Última verificación: **7-sep-2026**, antes de migrar del PC Lenovo a un MSI.

## Estado funcional

Clasificador supervisado de noticias en español en **12 categorías**, con
pipeline TF-IDF + regresión logística. Trabajo del curso **"Despliegue de
soluciones" (MAIA, Universidad de los Andes)**; el repositorio pertenece a la
organización `DesarolloSolucionesMaia`, no a la cuenta personal.

- `pytest -q` → **10 pasan, 0 fallan** (12 s).
- Modelo entrenado y evaluado. Métricas en `reports/metrics.json`:

| métrica | valor |
|---|---|
| accuracy | 0,9220 |
| macro F1 | 0,9221 |
| macro precision | 0,9263 |
| macro recall | 0,9223 |
| top-3 accuracy | 0,9878 |
| log loss | 0,8575 |

- Rama de trabajo: **`feature/logistic-regression-classifier`**, que además es
  la rama por defecto del remoto. No hay `main`.
- **Repo compartido.** El compañero jgarciaoUniandes añadió el 2-sep-2026
  (`89178fa`) el seguimiento con **MLflow** (`src/experiments.py`), con dos
  ejecuciones registradas en el experimento `spanish-news-classification`:

| Run | Configuración | Macro F1 | Accuracy | Log loss |
|---|---|---:|---:|---:|
| `logistic_regression_baseline` | unigramas, 30.000 features, C=1.0 | 0,9156 | 0,9146 | 1,0134 |
| **`logistic_regression_improved`** | uni+bigramas, 50.000 features, C=2.0 | **0,9221** | **0,9220** | **0,8575** |

  La ejecución seleccionada es `logistic_regression_improved`; es la que
  corresponde a las métricas de `reports/metrics.json` de arriba.

## Arquitectura

- `src/data_validation.py` — comprueba columnas, nulos, textos vacíos,
  duplicados, presencia de las 12 etiquetas, distribución y longitudes
- `src/explore.py` — exploración reproducible; genera la evidencia por clase
- `src/train.py` — partición estratificada 80/20, entrena y guarda artefactos
- `src/predict.py` — inferencia
- `tests/` — 10 pruebas (validación, entrenamiento, predicción)
- `artifacts/spanish_news_classifier.joblib` — **no se versiona**; se registrará
  después vía MLflow
- `reports/` — métricas, matriz de confusión, distribución de clases, perfiles
  y ejemplos por clase

## Comandos

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python .\src\data_validation.py    # validar el dataset
python .\src\explore.py            # exploración reproducible
python .\src\train.py              # entrenar (genera artifacts/ y reports/)
python .\src\predict.py            # inferencia
python -m pytest -q                # 10 pruebas
```

## Dependencias externas

- **Repositorio hermano `pdf-data`**, que administra el dataset con **DVC**.
  Ruta esperada, relativa a este repo:
  `../pdf-data/data/raw/spanish-news-classification/data/train-00000-of-00001.parquet`
- Fuente del dataset: `mteb/SpanishNewsClassification`,
  revisión `9f568e396857395bd803c63452b83316124d31c9`
- scikit-learn, pandas, joblib (ver `requirements.txt`)
- **MLflow 3.15.2** — integrado (`src/experiments.py`). `mlflow.db` y `mlruns/`
  son locales y **no** están en Git

## Variables de entorno

Ninguna. No hay secretos en el repo y no deben añadirse.

## Errores y limitaciones conocidas

1. **Depende de que `pdf-data` esté clonado como hermano** en el directorio
   padre. Sin esa ruta relativa exacta, ni la validación ni el entrenamiento
   funcionan. Es la trampa principal al montar el proyecto en otro equipo.
2. **`artifacts/` no está versionado**: el `.joblib` hay que regenerarlo con
   `train.py` en cada máquina nueva.
3. **`mlflow.db` y `mlruns/` no se versionan**: el historial de experimentos es
   local de cada máquina y hay que regenerarlo con `src/experiments.py`.
4. La rama por defecto es `feature/logistic-regression-classifier`; no existe
   `main`. Cuidado al asumir lo contrario en scripts o CI.
5. **Es un repo de equipo.** Este clon local llegó a estar divergido del remoto
   (un commit propio sin subir y uno del compañero sin bajar). Hacer `git fetch`
   antes de dar por bueno el estado o de declarar tareas pendientes.

## Último trabajo realizado

Sobre el commit `4c0f619` ("model: TF-IDF logistic regression classifier
implementado"), quedó sin commitear una tanda de trabajo real: `src/explore.py`
completo, el directorio `tests/` entero (10 pruebas), cuatro salidas nuevas en
`reports/` y actualizaciones de `README.md` y `requirements.txt`. Se commitea el
7-sep-2026 en el cierre para la migración.
