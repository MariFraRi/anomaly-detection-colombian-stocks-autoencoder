# **Identificación de Comportamientos Anómalos en Mercados Financieros: Un Enfoque Basado en Deep Learning y Series Temporales de ADRs**

## **Descripción**

Este proyecto implementa un enfoque de Deep Learning no supervisado para detectar anomalías puntuales en ADRs colombianos listados en la Bolsa de Nueva York (NYSE). En lugar de predecir precios, se emplea un **Denoising Autoencoder (DAE) basado en arquitectura GRU** para aprender el comportamiento normal del mercado y detectar desviaciones significativas mediante el error de reconstrucción (MSE).

El modelo aprende representaciones latentes del comportamiento histórico normal y señala como anomalía cualquier ventana temporal cuyo error de reconstrucción supere un umbral estadístico calibrado sobre datos de validación.

## **Objetivo**

Desarrollar una herramienta de alerta temprana capaz de identificar cambios estructurales y anomalías puntuales en el mercado financiero colombiano, utilizando exclusivamente información histórica de mercado y sin etiquetas supervisadas.

## **Activos analizados**

| Ticker | Empresa     | Sector       |
|--------|-------------|--------------|
| EC     | Ecopetrol   | Energía      |
| CIB    | Bancolombia | Financiero   |
| AVAL   | Grupo Aval  | Financiero   |
| TGLS   | Tecnoglass  | Manufactura  |

Datos diarios extraídos de **Yahoo Finance** para el período **2015–2024**.

## **Metodología**

**Partición cronológica (sin fuga de datos):**
- Entrenamiento: 2015–2019
- Validación: 2020 (choque COVID-19)
- Prueba: 2021–2024

**Feature Engineering:**
- `log_return`: retorno logarítmico diario
- `vol_21d`: volatilidad realizada (ventana de 21 sesiones)
- `vol_zscore`: Z-score del volumen transado

**Configuración del modelo:**
- Ventana temporal: T = 30 sesiones
- Tensor de entrada: `(batch, 30, 3)`
- Un modelo por activo
- Normalización: `StandardScaler` ajustado exclusivamente sobre el conjunto de entrenamiento
- Umbral de anomalía: percentil 75 del error MSE en validación

**Modelos benchmark (NB4):**
- Z-Score
- Isolation Forest
- One-Class SVM
- LSTM Predictor
- GRU Predictor
- DAE-LSTM
- DAE-GRU *(ganador en 4/4 activos)*

**Modelo principal:** DAE-GRU

## **Estructura del proyecto**
anomaly-detection-colombian-stocks-autoencoder/
│
├── data/
│   ├── raw/                  # Datos crudos descargados de Yahoo Finance
│   └── processed/            # Tensores .npy y scalers .pkl exportados desde NB2
│
├── notebooks/
│   ├── NB0_context.md        # Descripción del proyecto
│   ├── NB1_EDA.ipynb         # Análisis exploratorio y calidad de datos
│   ├── NB2_features.ipynb    # Feature engineering y exportación de artefactos
│   ├── NB3_DAE.ipynb         # Entrenamiento del Denoising Autoencoder
│   ├── NB4_benchmarks.ipynb  # Comparación con modelos benchmark
│   ├── NB5_evaluation.ipynb  # Evaluación final del modelo propuesto
│   └── NB6_tests.ipynb       # Tests estadísticos de significancia
│
├── models/                   # Pesos entrenados por activo
└── reports/                  # Figuras, tablas y resultados exportados
## **Tecnologías**

- Python 3.x
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- Pandas / NumPy
- Plotly / Matplotlib
- yfinance

## **Integrantes**

- Mariana Franco
- Danier Conde

## **Resultados**

El modelo DAE-GRU superó a los 6 modelos benchmark en los 4 activos evaluados, con una AUC-PR de **0.6048** (posición 2, activo EC) como referencia de validación. El sistema detecta correctamente períodos de estrés documentados: crisis del petróleo (2015–2016), colapso por COVID-19 (febrero–mayo 2020) y el ciclo de alza de tasas de la Fed (2022).