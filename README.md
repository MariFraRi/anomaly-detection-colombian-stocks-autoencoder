# Proyecto_DL
=======
# **Anomaly Detection in Colombian Stocks using Autoencoders**

## **Descripción**

Este proyecto implementa un enfoque de Deep Learning no supervisado para detectar anomalías y cambios de régimen en acciones colombianas.

En lugar de predecir precios, se utiliza un Denoising Autoencoder basado en LSTM/GRU para aprender el comportamiento normal del mercado y detectar desviaciones mediante el error de reconstrucción.

## **Objetivo**

Desarrollar una herramienta de alerta temprana para identificar cambios estructurales en el mercado financiero colombiano.

## **Metodología**

* Series de tiempo financieras (2015–2024)
* Feature engineering:

  * Retornos logarítmicos
  * Volatilidad realizada
  * Z-score del volumen
* Modelos benchmark
* Autoencoder (modelo principal)

## **Estructura del proyecto**

```
data/
notebooks/
models/
reports/
```

## **Tecnologías**

* Python
* TensorFlow / PyTorch
* Scikit-learn
* Pandas / NumPy

## **Resultados esperados**

Detección de anomalías y cambios de régimen en el comportamiento del mercado.

