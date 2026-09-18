# DECISIONS — model

### D-01 — El dataset no vive aquí: lo administra `pdf-data` con DVC
Separar datos de código es el punto del taller. La ruta esperada es relativa
(`../pdf-data/data/raw/...`), así que **los dos repos deben ser hermanos** en el
mismo directorio padre.

### D-02 — La revisión del dataset se fija explícitamente
`9f568e396857395bd803c63452b83316124d31c9` de `mteb/SpanishNewsClassification`.
Sin fijar la revisión, las métricas dejan de ser comparables entre corridas.

### D-03 — `artifacts/` no se versiona
El `.joblib` se regenera con `train.py`. Se registrará vía MLflow más adelante;
meter binarios de modelo en Git es justo lo que el curso enseña a no hacer.

### D-04 — Partición estratificada 80/20
Con 12 clases y distribución desigual, una partición al azar puede dejar clases
mal representadas en prueba.

### D-05 — La validación de datos es un paso propio, antes de entrenar
`data_validation.py` comprueba columnas, nulos, textos vacíos, duplicados,
presencia de las 12 etiquetas, distribución y longitudes. Entrenar sobre datos
sin validar produce métricas que no significan nada.

### D-06 — La exploración genera evidencia en disco, no gráficos efímeros
`explore.py` escribe `data_profile_by_class.csv`, `data_examples.csv`,
`class_distribution.png` y `text_length_by_class.png`. La evidencia se entrega.

### D-07 — La rama de trabajo es `feature/logistic-regression-classifier`
Es también la rama por defecto del remoto. **No existe `main`.**
