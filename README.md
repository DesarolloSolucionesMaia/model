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