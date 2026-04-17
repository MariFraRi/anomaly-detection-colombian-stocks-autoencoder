# Notebook 0: Contexto y Metodología

**Proyecto:** Detección de Anomalías y Cambios de Régimen en Acciones Colombianas
mediante Denoising Autoencoders con Arquitecturas LSTM/GRU

**Autores:** Mariana Franco - Danier Conde - Samuel Bermúdez 
**Institución:** Universidad del Norte  
**Fecha:** 2026 

---

## Tabla de Contenido

1. Contexto del Problema
2. Planteamiento del Problema
3. Justificación del Uso de Deep Learning
4. Enfoque Metodológico
5. Fundamentos Teóricos: Autoencoders
6. Modelo Propuesto: Denoising Autoencoder con LSTM/GRU
7. Dataset y Periodo de Estudio
8. Variables Utilizadas
9. Diseño Experimental
10. Objetivos del Proyecto

---

## 1. Contexto del Problema

### **1.1 Mercados Financieros como Sistemas Complejos No Estacionarios**

Los mercados financieros constituyen sistemas dinámicos de alta dimensionalidad cuyo
comportamiento estadístico evoluciona de manera no estacionaria a lo largo del tiempo.
A diferencia de los procesos estocásticos estacionarios, en los que los momentos
estadísticos (media, varianza, covarianza) permanecen constantes, las series de precios
y retornos financieros exhiben propiedades que cambian estructuralmente en función de
las condiciones macroeconómicas, el sentimiento del mercado y eventos exógenos de
naturaleza sistémica.

Este fenómeno se manifiesta en tres dimensiones fundamentales:

- **Heterocedasticidad condicional:** La varianza de los retornos no es constante en
  el tiempo. Los periodos de alta volatilidad tienden a agruparse (efecto ARCH/GARCH),
  generando regímenes diferenciados de riesgo que no pueden modelarse adecuadamente
  mediante distribuciones homocedásticas.

- **Colas pesadas (leptocurtosis):** La distribución empírica de los retornos
  financieros presenta exceso de curtosis positivo respecto a la distribución normal,
  lo que implica que los eventos extremos ocurren con una frecuencia sustancialmente
  mayor a la predicha por modelos gaussianos.

- **Cambios de régimen:** Los mercados atraviesan periodos alternos de comportamiento
  estructuralmente distinto — periodos de baja volatilidad y tendencia alcista
  ("bull markets") versus periodos de alta volatilidad y caídas pronunciadas
  ("bear markets" o crisis) — que no pueden ser capturados por un único modelo
  paramétrico estático.

### **1.2 El Mercado Colombiano: Características Específicas**

El mercado bursátil colombiano, representado principalmente por la Bolsa de Valores
de Colombia (BVC), presenta características particulares que incrementan la complejidad
del análisis:

- **Baja liquidez relativa:** Comparado con mercados desarrollados, el mercado
  colombiano exhibe menores volúmenes de negociación, lo que amplifica el impacto
  de eventos puntuales sobre los precios.

- **Alta dependencia de factores macroeconómicos sectoriales:** Activos como
  Ecopetrol están estructuralmente acoplados al precio internacional del petróleo
  (WTI/Brent), generando correlaciones externas que introducen volatilidad
  importada.

- **Exposición a riesgo político y cambiario:** Las acciones colombianas incorporan
  primas de riesgo asociadas a la incertidumbre política local y a la volatilidad
  del peso colombiano frente al dólar estadounidense, factores que generan shocks
  abruptos y difícilmente predecibles.

- **Periodos de estrés identificables:** El periodo 2015–2024 comprende al menos
  tres eventos de estrés sistémico mayor: la crisis del precio del petróleo
  (2015–2016), la pandemia por COVID-19 (2020) y el ciclo de alzas de tasas de
  interés de la Reserva Federal (2022), todos con impacto documentado sobre el
  mercado colombiano.

---

## 2. Planteamiento del Problema

### **2.1 Definición del Problema**

El problema central de esta investigación se define como sigue:

> **Dado un conjunto de series de tiempo financieras multivariadas correspondientes
> a acciones colombianas, ¿es posible construir un sistema no supervisado capaz de
> aprender la representación latente del comportamiento normal del mercado y señalar
> automáticamente las observaciones que se desvían significativamente de dicha
> representación?**

Este problema se enmarca en la categoría de **detección de anomalías en series
temporales** (time series anomaly detection), un área de investigación activa en
el cruce entre el aprendizaje profundo y las finanzas cuantitativas.

### **2.2 Distinción Respecto a la Predicción de Precios**

Es fundamental establecer la diferencia conceptual entre el presente enfoque y el
problema clásico de predicción de precios:

| Dimensión | Predicción de Precios | Detección de Anomalías |
|---|---|---|
| **Objetivo** | Estimar P_{t+h} | Identificar si r_t es anómalo |
| **Supervisión** | Supervisada (etiqueta: precio futuro) | No supervisada (sin etiquetas) |
| **Métrica** | RMSE, MAE sobre predicciones | MSE de reconstrucción, AUC |
| **Horizonte** | Futuro (h pasos) | Presente / pasado reciente |
| **Asunción** | Patrones históricos predicen el futuro | El presente desvía de lo "normal" |
| **Utilidad práctica** | Trading algorítmico | Gestión de riesgo, alerta temprana |

La predicción de precios enfrenta el problema fundamental de la hipótesis de mercados
eficientes (EMH), que argumenta que los precios incorporan toda la información
disponible, haciendo que los retornos futuros sean esencialmente impredecibles. La
detección de anomalías, en cambio, no requiere que los precios sean predecibles: sólo
requiere que el comportamiento "normal" sea estadísticamente distinguible del
comportamiento anómalo, lo cual es una condición considerablemente menos restrictiva.

### **2.3 Relevancia Práctica**

La detección de anomalías en series financieras tiene aplicaciones directas en:

- **Gestión de riesgo:** Identificación temprana de cambios de régimen que justifiquen
  el ajuste dinámico de posiciones o la activación de coberturas.
- **Vigilancia de mercados:** Detección de manipulación de precios o comportamientos
  anómalos de volumen que puedan señalar actividad irregular.
- **Construcción de portafolios:** Identificación de periodos de correlación anómala
  entre activos, relevante para la diversificación dinámica.
- **Backtesting y validación de modelos:** Detección de quiebres estructurales que
  invaliden los supuestos de estacionariedad en modelos de riesgo (VaR, CVaR).

---

## 3. Justificación del Uso de Deep Learning

### **3.1 Limitaciones de los Enfoques Clásicos**

Los métodos tradicionales de detección de anomalías en series de tiempo presentan
limitaciones estructurales cuando se aplican a datos financieros:

**Métodos estadísticos paramétricos (Z-score, IQR):** Asumen distribuciones
específicas (frecuentemente gaussianas) que no se ajustan a la distribución empírica
de los retornos financieros. La detección basada en umbrales estáticos no se adapta
a los cambios de régimen de volatilidad.

**Modelos ARIMA/GARCH:** Si bien capturan la estructura de autocorrelación y la
heterocedasticidad condicional, son modelos univariados que no explotan la estructura
de dependencia cruzada entre activos. Además, requieren re-estimación periódica ante
quiebres estructurales.

**Isolation Forest y One-Class SVM:** Los métodos de detección de anomalías clásicos
del aprendizaje automático operan sobre vectores de características independientes
(i.i.d.), ignorando la naturaleza secuencial y la dependencia temporal de las series
financieras. La detección de anomalías en el contexto temporal requiere que el modelo
evalúe una ventana de observaciones como una unidad, no puntos aislados.

### **3.2 Ventajas del Deep Learning para Series Temporales Financieras**

Las arquitecturas de aprendizaje profundo ofrecen capacidades específicas que
justifican su uso en este problema:

- **Aprendizaje de representaciones no lineales:** Las redes neuronales profundas
  pueden aprender transformaciones no lineales de los datos de entrada, capturando
  dependencias complejas entre variables y a través del tiempo que los modelos
  lineales no pueden representar.

- **Modelado de dependencias temporales de largo alcance:** Las arquitecturas
  recurrentes (LSTM, GRU) están diseñadas específicamente para capturar dependencias
  secuenciales, superando la limitación de los modelos ARIMA que sólo consideran
  un número fijo y pequeño de lags.

- **Aprendizaje multivariado nativo:** Un único modelo puede procesar
  simultáneamente múltiples series (retorno, volatilidad, volumen de varios activos),
  aprendiendo la estructura de co-movimiento normal y detectando anomalías de
  correlación cruzada.

- **Detección no paramétrica:** Al no asumir una distribución específica de los
  datos, el modelo aprende la geometría real del espacio de comportamientos normales
  directamente de los datos históricos.

- **Escalabilidad:** Una vez entrenado, el modelo produce una puntuación de anomalía
  (error de reconstrucción) en tiempo constante, independientemente de la complejidad
  del patrón aprendido.

---

## 4. Enfoque Metodológico

### **4.1 Aprendizaje No Supervisado**

Este proyecto adopta un enfoque estrictamente **no supervisado**. La motivación
principal es la ausencia de etiquetas de anomalía confiables y exhaustivas en datos
financieros históricos reales. Si bien es posible identificar retrospectivamente
algunos periodos de crisis (COVID-19, crisis del petróleo), las siguientes
consideraciones justifican el enfoque no supervisado:

1. **Escasez de etiquetas:** Los eventos de anomalía genuina son, por definición,
   poco frecuentes. Un clasificador supervisado entrenado con tan pocos ejemplos
   positivos sufriría de desbalance de clases severo y generalización deficiente.

2. **Heterogeneidad de anomalías:** No todas las anomalías siguen el mismo patrón.
   Un modelo no supervisado puede detectar anomalías con perfiles estadísticos
   desconocidos (out-of-distribution), mientras que un clasificador supervisado
   sólo detecta anomalías similares a las vistas durante el entrenamiento.

3. **Evolución del mercado:** Los patrones de anomalía cambian con el tiempo.
   Un modelo no supervisado que aprende la distribución normal vigente es más
   robusto a esta deriva que un clasificador entrenado sobre crisis históricas
   específicas.

### **4.2 Principio de Operación**

El principio fundamental del enfoque propuesto puede enunciarse como:

> **El modelo aprende exclusivamente sobre datos correspondientes al comportamiento
> normal del mercado. Cualquier secuencia que el modelo no pueda reconstruir con
> fidelidad — medida por el error cuadrático medio (MSE) — se considera
> potencialmente anómala.**

Este principio implica que la calidad de la detección depende críticamente de:

- La correcta definición y delimitación del conjunto de entrenamiento (sólo
  comportamiento normal).
- La capacidad del cuello de botella (espacio latente) para representar fielmente
  la estructura normal sin sobre-ajustarse a ruido.
- La elección del umbral de detección sobre el error de reconstrucción.

---

## 5. Fundamentos Teóricos: Autoencoders

### **5.1 Arquitectura General**

Un autoencoder es una red neuronal diseñada para aprender una representación
comprimida (codificación) de los datos de entrada mediante dos componentes
funcionalmente distintos:

**Encoder (codificador):** Una función parametrizada f_θ que mapea la entrada
x ∈ R^n a una representación latente z ∈ R^d, donde d << n:

```
z = f_θ(x)
```

**Decoder (decodificador):** Una función parametrizada g_φ que reconstruye la
entrada original a partir de la representación latente:

```
x̂ = g_φ(z)
```

El entrenamiento minimiza el error de reconstrucción entre la entrada original x
y su reconstrucción x̂:

```
L(θ, φ) = ||x - g_φ(f_θ(x))||²
         = MSE(x, x̂)
```

### **5.2 El Cuello de Botella como Regularización**

La restricción dimensional z ∈ R^d (con d << n) fuerza al modelo a aprender
una representación comprimida que captura únicamente la estructura esencial de
los datos. Esta compresión actúa como una forma implícita de regularización:

- **Memorización imposible:** El modelo no puede memorizar observaciones
  individuales; debe aprender patrones generalizables.
- **Eliminación de ruido:** Las dimensiones del espacio latente capturan
  variaciones de alta varianza (patrones recurrentes), mientras que el ruido
  de baja varianza no puede ser codificado eficientemente.
- **Compacidad de la distribución normal:** Los datos normales, al compartir
  estructura estadística, se proyectan en regiones densas del espacio latente.
  Las anomalías, al ser estadísticamente distintas, se proyectan fuera de estas
  regiones y generan reconstrucciones de alta pérdida.

### **5.3 Autoencoders como Detectores de Anomalías**

El error de reconstrucción MSE_t = ||x_t - x̂_t||² constituye la puntuación
de anomalía del modelo. El mecanismo de detección opera como sigue:

1. El modelo se entrena sobre secuencias normales → aprende a reconstruirlas
   con error bajo.
2. Ante una secuencia anómala, la representación latente z no corresponde a
   ningún patrón visto durante el entrenamiento.
3. El decoder, condicionado sobre una representación latente "errónea",
   produce una reconstrucción x̂ que difiere significativamente de x.
4. MSE_t supera el umbral τ → la secuencia se clasifica como anómala.

```
Anomalía si:  MSE_t > τ
Normal si:    MSE_t ≤ τ
```

donde τ se determina empíricamente como un percentil alto (p.ej., percentil 95
o 99) de la distribución de errores de reconstrucción sobre el conjunto de
entrenamiento.

---

## 6. Modelo Propuesto: Denoising Autoencoder con LSTM/GRU

### **6.1 Denoising Autoencoder (DAE)**

Un Denoising Autoencoder (Vincent et al., 2008) extiende el autoencoder estándar
introduciendo ruido artificial en la entrada durante el entrenamiento, mientras
el objetivo de reconstrucción permanece siendo la entrada original limpia:

```
x̃ = x + ε,    ε ~ N(0, σ²_noise)

L_DAE(θ, φ) = ||x - g_φ(f_θ(x̃))||²
```

Esta modificación produce tres beneficios fundamentales para el problema en cuestión:

- **Mayor robustez:** El modelo aprende representaciones invariantes a perturbaciones
  menores, evitando que el ruido de mercado de corto plazo active falsamente el
  detector.
- **Regularización implícita:** Al entrenar con versiones ruidosas, el modelo es
  forzado a capturar la estructura subyacente (el "manifold" de comportamiento normal)
  en lugar de memorizar las observaciones exactas.
- **Umbral de detección más limpio:** La distribución del error de reconstrucción
  en el conjunto de entrenamiento se comprime (menor varianza), produciendo un umbral
  τ más preciso y menos susceptible a falsos positivos.

### **6.2 Arquitecturas Recurrentes: LSTM y GRU**

La naturaleza secuencial de las series de tiempo financieras requiere que el
encoder y decoder sean capaces de modelar dependencias temporales. Se consideran
dos arquitecturas recurrentes:

**Long Short-Term Memory (LSTM):**
El LSTM (Hochreiter & Schmidhuber, 1997) introduce un mecanismo de memoria
controlada mediante tres compuertas (olvidar, entrada, salida) y una célula
de memoria separada del estado oculto. Esta separación permite al modelo
mantener información relevante a lo largo de secuencias largas sin sufrir
el problema del desvanecimiento del gradiente.

```
Compuertas LSTM:
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t] + b_C)
  h_t = o_t ⊙ tanh(C_t)
```

**Gated Recurrent Unit (GRU):**
El GRU (Cho et al., 2014) simplifica el LSTM fusionando las compuertas de olvidar
y entrada en una única compuerta de actualización, y eliminando la célula de
memoria separada. Produce resultados comparables al LSTM con un menor número de
parámetros, lo cual es ventajoso cuando el tamaño del conjunto de entrenamiento
es limitado.

```
Compuertas GRU:
  z_t = σ(W_z · [h_{t-1}, x_t])    # Update gate
  r_t = σ(W_r · [h_{t-1}, x_t])    # Reset gate
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### **6.3 Arquitectura del Modelo Propuesto**

La arquitectura del Denoising Autoencoder recurrente se define como sigue:

```
ENTRADA: tensor de forma (batch_size, T, F)
         T = longitud de secuencia (ventana temporal, justificada por ACF/PACF)
         F = número de features por timestep

ENCODER:
  Capa 1: LSTM/GRU(units=64, return_sequences=True)  + Dropout(0.2)
  Capa 2: LSTM/GRU(units=32, return_sequences=False) + Dropout(0.2)
  Capa 3: Dense(units=latent_dim)    # Cuello de botella (espacio latente z)

DECODER:
  Capa 4: RepeatVector(T)            # Replica z para cada timestep
  Capa 5: LSTM/GRU(units=32, return_sequences=True) + Dropout(0.2)
  Capa 6: LSTM/GRU(units=64, return_sequences=True) + Dropout(0.2)
  Capa 7: TimeDistributed(Dense(F))  # Reconstrucción por timestep

SALIDA: tensor de forma (batch_size, T, F)
        Objetivo: minimizar MSE(entrada_limpia, reconstrucción)
```

**Parámetros a determinar experimentalmente:**

- Dimensión del espacio latente (`latent_dim`): se evaluarán valores en {8, 16, 32}
- Longitud de ventana T: justificada mediante ACF/PACF (Notebook 2)
- Nivel de ruido del DAE (σ_noise): se evaluarán valores en {0.01, 0.05, 0.1}
- Tasa de dropout: se evaluarán valores en {0.1, 0.2, 0.3}

---

## 7. Dataset y Periodo de Estudio

### **7.1 Activos Seleccionados**

| Ticker | Empresa | Sector | Justificación |
|---|---|---|---|
| EC | Ecopetrol S.A. | Energía / Petróleo | Empresa más líquida del mercado colombiano; alta sensibilidad a shocks externos |
| CIB | Bancolombia S.A. | Banca | Representativo del sector financiero; sensible a política monetaria |
| AVAL | Grupo Aval | Conglomerado financiero | Exposición diversificada al sector bancario y de infraestructura |
| TGLS | Tecnoglass Inc. | Industrial | Empresa colombiana listada en NASDAQ; exposición diferenciada |

> **Nota:** Los activos se obtienen en su versión ADR (American Depositary Receipt)
> desde Yahoo Finance, lo que garantiza datos en USD con cobertura diaria continua
> sin interrupciones por festivos locales colombianos.

### **7.2 Periodo de Estudio**

El periodo de análisis comprende desde el 1 de enero de 2015 hasta el 31 de
diciembre de 2024, totalizando aproximadamente 2,500 sesiones de negociación
por activo. Este periodo fue seleccionado por las siguientes razones:

- **Suficiencia estadística:** Un mínimo de 2,000 observaciones es necesario para
  estimar distribuciones empíricas robustas del error de reconstrucción.
- **Cobertura de regímenes múltiples:** El periodo incluye mercados alcistas
  (2016–2019, 2021), crisis sistémicas (2020) y ciclos de contracción monetaria
  (2022–2023), proporcionando diversidad de regímenes para entrenamiento y evaluación.
- **Relevancia temporal:** La inclusión del periodo post-COVID permite evaluar el
  comportamiento del modelo ante un nuevo régimen de mercado tras una crisis sin
  precedentes.

### **7.3 Periodos de Referencia para Evaluación**

Los siguientes periodos constituyen la referencia de anomalías conocidas para la
validación cualitativa del modelo:

| Periodo | Fechas | Descripción | Activos más afectados |
|---|---|---|---|
| Crisis del petróleo | Jul 2015 – Feb 2016 | Desplome del precio del crudo WTI | EC |
| Pandemia COVID-19 | Feb 2020 – May 2020 | Shock sistémico global | Todos |
| Recuperación post-COVID | Jun 2020 – Dic 2020 | Régimen de alta volatilidad positiva | Todos |
| Ciclo de alzas Fed | Ene 2022 – Dic 2022 | Endurecimiento monetario global | Todos |
| Incertidumbre política Colombia | May 2021 – Ene 2022 | Elecciones y paro nacional | CIB, AVAL |

---

## 8. Variables Utilizadas

### **8.1 Variable Principal: Retorno Logarítmico**

El retorno logarítmico diario se define como:

```
r_t = ln(P_t / P_{t-1})
```

**Justificación:**

- A diferencia del retorno aritmético, el retorno logarítmico es aditivo en el
  tiempo, lo que facilita la agregación de retornos multi-periodo.
- Las series de retorno logarítmico son estacionarias en primer orden (confirmado
  mediante el test ADF en Notebook 1), a diferencia de los precios de cierre que
  exhiben raíz unitaria.
- La escala compacta de los retornos logarítmicos (valores típicamente en [-0.10,
  0.10]) reduce los problemas de escala numérica durante el entrenamiento.

### **8.2 Variable de Volatilidad: Volatilidad Rodante**

La volatilidad realizada en una ventana de 21 días de negociación (aproximadamente
un mes calendario) se define como:

```
σ_t = std(r_{t-20}, r_{t-19}, ..., r_t) × √252
```

donde la multiplicación por √252 anualiza la medida.

**Justificación:**

- Captura el nivel de riesgo condicional local, proveyendo al modelo información
  sobre el régimen de volatilidad vigente.
- Al ser una transformación de los retornos (diferencias), la serie resultante es
  estacionaria.
- La ventana de 21 días refleja el horizonte mensual estándar utilizado en gestión
  de riesgos (Value at Risk mensual).

### **8.3 Variable de Actividad: Volumen Z-Score**

El volumen de negociación normalizado se define como:

```
z_vol_t = (log(V_t) - μ_V(21d)) / σ_V(21d)
```

donde μ_V(21d) y σ_V(21d) son la media y desviación estándar del log-volumen en
la ventana rodante de 21 días inmediatamente anteriores a t.

**Justificación:**

- El volumen bruto no es comparable entre activos ni a lo largo del tiempo
  (tendencia secular al alza en la mayoría de mercados). La normalización
  z-score elimina esta no-estacionariedad de nivel.
- Los eventos anómalos frecuentemente se acompañan de volúmenes atípicos
  (pánico vendedor, noticias sorpresivas), proveyendo una señal ortogonal
  a la dirección del retorno.
- La transformación logarítmica previa corrige la asimetría positiva inherente
  a las series de volumen.

### **8.4 Resumen del Vector de Features**

| Feature | Símbolo | Descripción | Periodicidad |
|---|---|---|---|
| Retorno logarítmico | r_t | ln(P_t / P_{t-1}) | Diaria |
| Volatilidad rodante (21d) | σ_t | Desv. estándar anualizada | Diaria (ventana 21d) |
| Volumen z-score (21d) | z_vol_t | Vol. normalizado respecto a media rodante | Diaria (ventana 21d) |

> **Nota sobre features adicionales:** En una extensión del modelo base se evaluará
> la inclusión de indicadores técnicos (RSI, bandas de Bollinger, MACD). Sin embargo,
> el modelo base se construye exclusivamente con las tres variables anteriores para
> mantener la parsimonia y facilitar la interpretación de los resultados.

---

## 9. Diseño Experimental

### **9.1 Partición Temporal de los Datos**

La partición de los datos sigue un esquema **estrictamente cronológico**, sin
asignación aleatoria, para evitar el data leakage temporal — la contaminación del
conjunto de entrenamiento con información estadística del futuro:

```
|─────────────────────────────|──────────────|──────────────────────────────|
| ENTRENAMIENTO               | VALIDACIÓN   | TEST                         |
| 2015-01-01 → 2019-12-31     | 2020-01-01   | 2021-01-01 → 2024-12-31      |
| (~1,258 sesiones)           | 2020-12-31   | (~1,000 sesiones)            |
|                             | (~252 ses.)  |                              |
|─────────────────────────────|──────────────|──────────────────────────────|
                              ↑              ↑
                        Split 1          Split 2
                   (pre-COVID)     (COVID completo
                                   en validación)
```

**Justificación de la partición:**

- **Entrenamiento (2015–2019):** Comprende el régimen de "comportamiento normal"
  dominante, con la excepción de la crisis del petróleo (2015–2016), que provee
  diversidad moderada de volatilidad sin contaminar el conjunto con el mayor
  evento de estrés del periodo (COVID-19).
- **Validación (2020):** Contiene el periodo COVID completo, el evento anómalo
  de referencia primaria. Se utiliza para seleccionar el umbral τ de detección y
  para el early stopping durante el entrenamiento.
- **Test (2021–2024):** Periodo completamente reservado para la evaluación final,
  nunca utilizado en decisiones de diseño o ajuste del modelo.

### **9.2 Generación de Ventanas Temporales**

El modelo recibe como entrada ventanas deslizantes de longitud T (a determinar
en Notebook 2 mediante ACF/PACF):

```python
# Pseudocódigo de generación de ventanas
for t in range(T, len(data)):
    ventana = data[t-T : t]        # forma: (T, F)
    X.append(ventana)              # sin etiquetas (aprendizaje no supervisado)
```

- Las ventanas se generan con stride = 1 (ventana deslizante de un día).
- No existe solapamiento entre las ventanas de entrenamiento y validación/test.
- La primera ventana válida inicia en el timestep T + warm_up_period,
  donde warm_up_period = max(21 días de volatilidad rodante, 26 días de MACD).

### **9.3 Protocolo de Evaluación**

La evaluación del modelo se realiza en dos niveles:

**Evaluación cuantitativa:**

- Distribución del error de reconstrucción MSE por split (entrenamiento vs.
  validación vs. test).
- Selección del umbral τ como el percentil p de la distribución MSE del conjunto
  de entrenamiento. Se evaluarán p ∈ {90, 95, 99}.
- Métricas de detección contra etiquetas de anomalía conocidas:
  Precisión, Recall, F1-Score, AUC-ROC.

**Evaluación cualitativa:**

- Visualización de la serie temporal del MSE con los periodos de crisis anotados.
- Verificación de que los picos de MSE coinciden con eventos anómalos conocidos.
- Análisis del espacio latente mediante reducción dimensional (t-SNE/UMAP)
  para confirmar la separabilidad de regímenes normales y anómalos.

### **9.4 Manejo del Data Leakage**

Se identifican y controlan explícitamente los siguientes vectores de data leakage:

| Vector de Leakage | Descripción | Control Aplicado |
|---|---|---|
| **Leakage de normalización** | El scaler aprende estadísticas del conjunto completo | Scaler ajustado exclusivamente sobre el conjunto de entrenamiento |
| **Leakage de ventana** | Ventanas que inician en entrenamiento y terminan en validación | Generación estricta de ventanas por split sin cruce de fronteras |
| **Leakage de features rodantes** | La volatilidad rodante incorpora observaciones futuras | Todas las features se computan con exclusividad hacia el pasado (causal) |
| **Leakage de umbral** | El umbral τ se selecciona sobre datos de test | τ se selecciona sobre el conjunto de validación, nunca sobre test |
| **Leakage de selección del modelo** | Los hiperparámetros se ajustan sobre datos de test | Toda búsqueda de hiperparámetros usa exclusivamente validación |

---

## 10. Objetivos del Proyecto

### **10.1 Objetivo General**

Desarrollar y evaluar un sistema de detección de anomalías y cambios de régimen
en series de tiempo financieras de acciones colombianas, basado en un Denoising
Autoencoder con arquitectura recurrente (LSTM/GRU), mediante un enfoque de
aprendizaje no supervisado que aprenda la representación latente del comportamiento
normal del mercado y señale desviaciones significativas a través del error de
reconstrucción.

### **10.2 Objetivos Específicos**

1. **Caracterizar estadísticamente** las series de tiempo de retornos, volatilidad
   y volumen de los activos colombianos seleccionados, identificando propiedades
   de estacionariedad, distribución y dependencia temporal que informen las
   decisiones de preprocesamiento y diseño arquitectónico.

2. **Diseñar y justificar** la ingeniería de features, la longitud de ventana
   temporal y la estrategia de partición de datos garantizando la ausencia de
   data leakage en todas sus formas.

3. **Implementar y entrenar** un Denoising Autoencoder con encoder y decoder
   recurrentes (LSTM y GRU), comparando ambas variantes en términos de capacidad
   de reconstrucción y sensibilidad detectora.

4. **Determinar empíricamente** el umbral óptimo de detección de anomalías sobre
   el conjunto de validación, analizando el trade-off entre precisión y recall para
   distintos niveles de percentil.

5. **Evaluar la capacidad del modelo** para identificar los periodos anómalos
   conocidos (crisis del petróleo, COVID-19, ciclo de alzas de tasas) mediante
   métricas cuantitativas y visualizaciones cualitativas del error de
   reconstrucción temporal.

6. **Analizar el espacio latente** aprendido por el encoder para determinar si
   las representaciones de regímenes normales y anómalos son geométricamente
   separables, validando la hipótesis de compresión selectiva del autoencoder.

---

## Referencias Metodológicas

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
  *Neural Computation*, 9(8), 1735–1780.

- Cho, K., et al. (2014). Learning phrase representations using RNN
  encoder-decoder for statistical machine translation. *EMNLP 2014*.

- Vincent, P., et al. (2008). Extracting and composing robust features
  with denoising autoencoders. *ICML 2008*.

- Malhotra, P., et al. (2016). LSTM-based encoder-decoder for multi-sensor
  anomaly detection. *ICML Anomaly Detection Workshop*.

- Fama, E. F. (1970). Efficient capital markets: A review of theory and
  empirical work. *The Journal of Finance*, 25(2), 383–417.

- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation.
  *Econometrica*, 50(4), 987–1007.

---

*Este notebook constituye el documento de referencia metodológica del proyecto.
Las decisiones de diseño aquí establecidas son vinculantes para todos los
notebooks de implementación subsiguientes.*
