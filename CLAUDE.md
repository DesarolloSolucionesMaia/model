# CLAUDE.md — model

Clasificador de noticias en español (12 categorías, TF-IDF + regresión
logística). Taller del curso **"Despliegue de soluciones"** (MAIA, Universidad
de los Andes). El repo pertenece a la organización `DesarolloSolucionesMaia`.

Antes de tocar nada:

1. `docs/PROJECT_STATE.md` — estado verificado, métricas reales, comandos.
2. `docs/DECISIONS.md` — las 7 decisiones vigentes.
3. `docs/NEXT_STEPS.md` — la siguiente tarea.
4. `README.md` — el detalle de cada paso del pipeline.

## Lo que te va a morder

- **El dataset no está en este repo.** Lo administra el repo hermano `pdf-data`
  con DVC, y la ruta es **relativa**:
  `../pdf-data/data/raw/spanish-news-classification/data/train-00000-of-00001.parquet`
  Si `pdf-data` no está clonado como hermano en el mismo directorio padre, ni la
  validación ni el entrenamiento funcionan.
- **La rama es `feature/logistic-regression-classifier`, no `main`.** También es
  la rama por defecto del remoto. No asumas `main` en scripts ni en CI.
- **`artifacts/` no se versiona.** El `.joblib` se regenera con `train.py`.
  Tampoco `mlflow.db` ni `mlruns/`: el historial de experimentos es local.
- **Es un repo de equipo** (organización `DesarolloSolucionesMaia`). Haz
  `git fetch` antes de dar por bueno el estado o de declarar algo pendiente:
  este clon ya llegó a estar divergido y una tarea "pendiente" resultó estar
  hecha por un compañero.

## Verificación antes de dar algo por bueno

```powershell
python -m pytest -q                # 10 esperadas
python .\src\data_validation.py    # antes de entrenar, siempre
```

## Criterio del curso

Validar los datos **antes** de entrenar, fijar la revisión del dataset, y dejar
la evidencia escrita en `reports/` en vez de gráficos que se pierden. Las
métricas solo son comparables si el dato de entrada es el mismo.

Sin secretos en el repo, y ninguno debe añadirse.
