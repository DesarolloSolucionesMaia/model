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

## API de inferencia

La API FastAPI carga el artefacto una sola vez al iniciar y expone:

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/health` | Verificar que el servicio está disponible. |
| `GET` | `/v1/model` | Consultar versión, métricas y metadatos del modelo. |
| `GET` | `/v1/categories` | Listar las 12 categorías para construir filtros del tablero. |
| `GET` | `/v1/predictions?limit=20&offset=0` | Consultar el historial paginado, del más reciente al más antiguo. |
| `GET` | `/v1/predictions/{prediction_id}` | Consultar el detalle de una predicción guardada. |
| `DELETE` | `/v1/predictions/{prediction_id}` | Eliminar una predicción del historial. |
| `POST` | `/v1/predictions` | Clasificar un texto. |
| `POST` | `/v1/predictions/batch` | Clasificar hasta 50 textos en una petición. |
| `POST` | `/v1/predictions/pdf` | Extraer el texto de un PDF de hasta 20 MB y clasificarlo. |

Cada predicción se almacena en SQLite y devuelve `prediction_id`,
`created_at`, `source_type` y, para PDFs, `file_name`. Por privacidad, la
base no almacena el texto ni el contenido del documento. El historial conserva
el resultado, top 3, versión temporal del modelo y un hash no reversible del
PDF. La ruta se puede configurar mediante `DATABASE_PATH`.

Ejecución sin Docker:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva:

```text
http://localhost:8000/docs
```

Ejemplo por texto:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/v1/predictions `
  -ContentType 'application/json' `
  -Body '{"text":"El equipo ganó la final con dos goles."}'
```

## Ejecución local con Docker

Requisitos: Docker Desktop iniciado y el archivo
`artifacts/spanish_news_classifier.joblib` presente.

Docker Compose crea el volumen `prediction-data`, por lo que el historial se
conserva aunque el contenedor sea recreado. `docker compose down` no elimina
este volumen; `docker compose down -v` sí lo elimina.

1. Ubicarse en el repositorio del modelo:

   ```powershell
   cd Repo\model
   ```

2. Construir e iniciar el contenedor:

   ```powershell
   docker compose up --build -d
   ```

3. Verificar la salud y abrir la documentación:

   ```powershell
   Invoke-RestMethod http://localhost:8000/health
   Start-Process http://localhost:8000/docs
   ```

4. Probar un PDF (PowerShell 7):

   ```powershell
   curl.exe -X POST http://localhost:8000/v1/predictions/pdf `
     -H "accept: application/json" `
     -F "file=@C:\ruta\noticia.pdf;type=application/pdf"
   ```

5. Consultar registros o detener el servicio:

   ```powershell
   docker compose logs -f api
   docker compose down
   ```

El origen del tablero se configura con `CORS_ORIGINS` en `compose.yaml`.
Para producción se debe reemplazar por el dominio real del frontend.

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
