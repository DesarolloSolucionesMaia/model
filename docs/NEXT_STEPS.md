# NEXT_STEPS — model

Estado al 7-sep-2026.

> Corregido el mismo día: la primera versión de este archivo ponía como
> siguiente tarea "registrar el modelo con MLflow". **Ya estaba hecho** por
> jgarciaoUniandes el 2-sep-2026 (`89178fa`), en commits que este clon local no
> se había traído. Lección: comprobar `git fetch` antes de declarar pendientes.

## Siguiente tarea concreta

**Revisar las clases con peor F1 en `reports/classification_report.csv`.**

El macro F1 de 0,9221 esconde qué categorías fallan. Con 12 clases y una
distribución desigual, es previsible que dos o tres carguen con casi todo el
error, y saberlo cambia qué se intenta después: si el fallo se concentra en
categorías semánticamente vecinas (Economía/Política, Motor/Tecnología), el
camino es de características; si se concentra en las clases con menos ejemplos,
el camino es de datos o de balanceo.

Hoy no se ha mirado, y sin eso cualquier "mejora" siguiente es a ciegas.

## Después

2. Comparar contra una línea base de verdad. Los dos runs registrados en MLflow
   (`logistic_regression_baseline` 0,9156 y `logistic_regression_improved`
   0,9221 de macro F1) son variantes del mismo pipeline TF-IDF + regresión
   logística. Falta un modelo de otra familia, o al menos un clasificador
   trivial, para saber si 0,92 es bueno en este dataset.
3. Empaquetado y despliegue, según lo que pida la guía del taller.

## Al llegar al MSI

- **Clonar `pdf-data` como repo hermano** en el mismo directorio padre, o nada
  funciona (D-01). Y correr `dvc pull` para bajar los datos.
- Recrear el `.venv`, nunca copiarlo.
- Regenerar `artifacts/spanish_news_classifier.joblib` con `python .\src\train.py`.
- `mlflow.db` y `mlruns/` son locales y no están en Git: los experimentos hay que
  volver a correrlos con `python .\src\experiments.py` si se quiere el historial.
- Ojo con la rama: es `feature/logistic-regression-classifier`, no `main`.
- **Es un repo compartido** (organización `DesarolloSolucionesMaia`, con al menos
  un compañero trabajando). Hacer `git fetch` antes de asumir el estado.
