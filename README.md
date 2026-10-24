# Analítica de accidentes de tránsito – Aplicación Streamlit (Entregable 5)

Aplicación del Proyecto Integrador de Analítica de Datos. Analiza las atenciones a víctimas de
accidentes de tránsito en 4 IPS de Bucaramanga (2018–2021) para:

* predecir qué pacientes requerirán cirugía (regresión logística, árbol de decisión, Random Forest y SVM);
* estimar el costo de la atención (regresión lineal múltiple sobre log10 del costo);
* segmentar pacientes (PCA y K-means).

**Fuente:** Datos Abiertos Colombia, conjunto `g4vd-w4ip` (https://www.datos.gov.co/resource/g4vd-w4ip.json),
limpiado en el Cuaderno 1 (`dataset_completo_limpio_transformado.csv`).

## Estructura
```
app.py              # aplicación
requirements.txt    # dependencias para Streamlit Community Cloud
data/dataset.csv    # dataset limpio del Cuaderno 1
README.md
```

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Módulos (barra lateral)
1. **Inicio**: contexto, decisión y cifras generales.
2. **Carga y análisis de datos**: carga de CSV, primeras 10 filas, describe(), info() y faltantes,
   histogramas, barras, matriz de correlación.
3. **Segmentación (PCA y K-means)**: scree plot, cargas, codo, silueta y perfiles por clúster.
4. **Regresión lineal del costo**: R², MAE, RMSE en pesos, real vs predicho, residuos y coeficientes.
5. **Modelos de clasificación**: selección del algoritmo, explicación, matriz de confusión,
   reporte de clasificación, curva ROC/AUC, curva Precision–Recall y umbral por costo de error.
6. **Predicciones y descarga**: tabla de predicciones descargable en CSV y predicción de un paciente nuevo.

## Reglas estadísticas
* Partición 80/20 estratificada **antes** de imputar, escalar o codificar (todo dentro de un `Pipeline`).
* Sin fuga: se excluyen identificadores, costos (en clasificación), condición de egreso, remisión,
  citas de control y variables de calendario (en clasificación).
* El umbral de cada modelo se elige en una validación interna del entrenamiento; la prueba solo se reporta.
* Semilla fija: 42.

## Despliegue en Streamlit Community Cloud
1. Subir esta carpeta a un repositorio público de GitHub.
2. Entrar a https://share.streamlit.io con la cuenta de GitHub → **Create app**.
3. Elegir el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings**, Python 3.11 → **Deploy**.
