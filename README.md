# Clasificador de noticias en español

Modelo supervisado para clasificar noticias en español en 12 categorías. El pipeline utiliza TF-IDF y regresión logística.

## Categorías

| Código | Categoría |
|---:|---|
| 0 | Alimentación |
| 1 | Astronomía |
| 2 | Economía |
| 3 | Moda |
| 4 | Medicina |
| 5 | Defensa |
| 6 | Motor |
| 7 | Entretenimiento |
| 8 | Política |
| 9 | Religión |
| 10 | Deportes |
| 11 | Tecnología |

## Dataset

El dataset se administra en el repositorio hermano `pdf-data` mediante DVC.

Ruta esperada:

```text
../pdf-data/data/raw/spanish-news-classification/data/train-00000-of-00001.parquet
```

Fuente: `mteb/SpanishNewsClassification`

Revisión utilizada:

```text
9f568e396857395bd803c63452b83316124d31c9
```

## Instalación

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validación de datos

```powershell
python .\src\data_validation.py
```

La validación comprueba:

- Columnas requeridas.
- Valores nulos.
- Textos vacíos.
- Textos duplicados.
- Presencia de las 12 etiquetas.
- Distribución de las categorías.
- Longitud de los textos.

## Entrenamiento

```powershell
python .\src\train.py
```

El proceso realiza una partición estratificada de 80 % para entrenamiento y 20 % para prueba.

Artefactos generados:

```text
artifacts/spanish_news_classifier.joblib
reports/metrics.json
reports/classification_report.csv
reports/confusion_matrix.png
```

El directorio `artifacts` no se almacena en Git. El modelo se registrará posteriormente mediante MLflow.

## Predicción

```powershell
python .\src\predict.py --text "Texto de una noticia en español"
```

La salida contiene:

- Categoría predicha.
- Confianza.
- Tres categorías más probables.

## Resultados de la primera versión

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9220 |
| Macro precision | 0.9263 |
| Macro recall | 0.9223 |
| Macro F1 | 0.9221 |
| Weighted F1 | 0.9220 |
| Log loss | 0.8575 |
| Top-3 accuracy | 0.9878 |

Configuración principal:

- TF-IDF con unigramas y bigramas.
- Máximo de 50.000 características.
- Regresión logística con `C=2.0`.
- Semilla reproducible `42`.

## Experimentos con MLflow

Los experimentos utilizan el mismo conjunto estratificado de entrenamiento y prueba para garantizar una comparación justa.

Experimento de MLflow:

```text
spanish-news-classification
```

Ejecuciones finales:

| Run | Configuración | Macro F1 | Accuracy | Log loss |
|---|---|---:|---:|---:|
| `logistic_regression_baseline` | Unigramas, 30.000 características, C=1.0 | 0.9156 | 0.9146 | 1.0134 |
| `logistic_regression_improved` | Unigramas y bigramas, 50.000 características, C=2.0 | 0.9221 | 0.9220 | 0.8575 |

La ejecución seleccionada es `logistic_regression_improved`, porque obtuvo mayor Macro F1 y accuracy, y menor log loss.

### Ejecutar los experimentos

```powershell
python .\src\experiments.py
```

Cada ejecución registra:

- Parámetros de TF-IDF y regresión logística.
- Accuracy, precision, recall y F1.
- Log loss y top-3 accuracy.
- Tiempos de entrenamiento e inferencia.
- Reporte de clasificación.
- Matriz de confusión.
- Homologación de etiquetas.
- Revisión del dataset.
- Commit Git.
- Modelo con firma y ejemplo de entrada.

### Abrir MLflow localmente

```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000
```

Interfaz:

```text
http://127.0.0.1:5000
```

La base `mlflow.db`, el directorio `mlruns` y los modelos generados son artefactos locales y no se almacenan directamente en Git.