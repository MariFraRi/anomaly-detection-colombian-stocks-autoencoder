# 3. Revisión del Estado del Arte

## 3.1 Estrategia de Búsqueda

**Palabras clave utilizadas**

La búsqueda bibliográfica se construyó mediante la combinación sistemática de los siguientes términos:

- `"anomaly detection" AND "time series" AND "deep learning"`
- `"LSTM autoencoder" AND "unsupervised" AND "financial time series"`
- `"reconstruction error" AND "anomaly detection" AND "recurrent neural network"`
- `"regime change detection" AND "deep learning" AND "equity markets"`
- `"encoder-decoder" AND "LSTM" AND "time series anomaly"`
- `"transformer" AND "anomaly detection" AND "multivariate time series"`

**Bases de datos consultadas**

- IEEE Xplore
- Scopus
- Web of Science (WoS) — categoría Q1: Computer Science, Artificial Intelligence
- ACM Digital Library
- arXiv (cs.LG, q-fin.ST) — como fuente complementaria para prepublicaciones relevantes

**Periodo de búsqueda**

2016–2024, con énfasis en contribuciones publicadas a partir de 2018. Se incluye el artículo seminal de Malhotra et al. (2016) como referencia fundacional del paradigma EncDec-AD.

**Criterios de inclusión**

- Uso explícito de Deep Learning (LSTM, GRU, Transformer, AE, VAE u híbridos).
- Abordaje del problema de detección de anomalías en series de tiempo (univariadas o multivariadas).
- Reporte de métricas cuantitativas de desempeño (F1, Precision, Recall, AUC-ROC).
- Publicación en venues arbitrados de alto impacto: ACM Computing Surveys, VLDB, KDD, ICLR, IEEE Transactions on Knowledge and Data Engineering (TKDE), IEEE Access, Information Sciences.

**Criterios de exclusión**

- Trabajos centrados exclusivamente en forecasting sin componente de detección de anomalías.
- Métodos supervisados con dependencia crítica de etiquetas densas (incompatibles con el marco no supervisado del proyecto).
- Artículos sin reporte de métricas comparables o sin código reproducible verificable.

---

## 3.2 Identificación del Top de Modelos

A partir de la revisión sistemática se identificaron **siete modelos** que satisfacen los criterios de inclusión y presentan relevancia directa para el problema de detección de anomalías y cambios de régimen en series de tiempo financieras. Los modelos se ordenan de acuerdo con su impacto arquitectónico y alineación con el presente proyecto:

| # | Modelo | Arquitectura | Venue | Año |
|---|--------|-------------|-------|-----|
| 1 | **EncDec-AD** | LSTM Encoder-Decoder (DAE) | ICML Workshop | 2016 |
| 2 | **OmniAnomaly** | VAE + Stochastic RNN | KDD | 2019 |
| 3 | **USAD** | Dual AE adversarial | KDD | 2020 |
| 4 | **Anomaly Transformer** | Transformer (self-attention) | ICLR | 2022 |
| 5 | **TranAD** | Transformer adversarial | VLDB | 2022 |
| 6 | **DeepAnT** | CNN predictor | IEEE Access | 2019 |
| 7 | **CAE-AD** | Contrastive LSTM AE | Information Sciences | 2022 |

---

## 3.3 Análisis Detallado por Modelo

---

### **Modelo 1: EncDec-AD — LSTM-based Encoder-Decoder for Anomaly Detection**

**Referencia completa**

> Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). *LSTM-based encoder-decoder for multi-sensor anomaly detection*. arXiv:1607.00148. Presentado en ICML 2016 Anomaly Detection Workshop.

**Tipo de arquitectura**

LSTM Encoder-Decoder (Recurrent Autoencoder). Aprendizaje no supervisado basado en reconstrucción.

**Descripción técnica**

El modelo aprende una representación comprimida de ventanas de series de tiempo normales mediante un encoder LSTM que mapea la secuencia de entrada $\mathbf{X}_{t-T+1:t} \in \mathbb{R}^{T \times d}$ a un vector de estado latente fijo $\mathbf{h} \in \mathbb{R}^{k}$. El decoder LSTM recibe $\mathbf{h}$ y reconstruye la secuencia de salida $\hat{\mathbf{X}}$. El error de reconstrucción $\text{MSE}(X, \hat{X})$ funciona como puntuación de anomalía: secuencias que no pertenecen al régimen normal producen errores sistemáticamente elevados. El umbral de detección se determina empíricamente sobre el conjunto de entrenamiento (distribución del error en datos normales).

**Innovación respecto a modelos previos**

A diferencia de los modelos LSTM de predicción puntual (LSTM-AD, Malhotra et al., 2015), EncDec-AD opera sobre la secuencia completa, haciéndolo robusto ante series de tiempo impredecibles donde el error de predicción forward no es informativo. Extiende los autoencoders denoising no temporales (Sakurada & Yairi, 2014) incorporando dependencias recurrentes.

**Tipo de problema**

Detección no supervisada de anomalías en series de tiempo univariadas y multivariadas, incluyendo series predecibles, periódicas, aperiódicas y cuasi-periódicas.

**Datasets utilizados**

Power Demand, Space Shuttle, ECG (públicos), Engine datasets con comportamiento predecible e impredecible (industriales, TCS Research).

**Métricas reportadas**

| Dataset | Precision | Recall | F$_{0.1}$ | TPR/FPR |
|---------|-----------|--------|-----------|---------|
| Engine-NP | 0.96 | 0.18 | 0.93 | 7.6 |
| Space Shuttle | — | — | 0.84 | — |
| Power Demand | — | — | 0.90 | — |

Sobre SMD (benchmark posterior): F1$_\text{best}$ = 0.7729 (Precision 0.9014, Recall 0.6764).

**Fortalezas**

- Robustez ante series impredecibles: no requiere que la serie sea pronosticable.
- Capacidad de manejar secuencias de longitud variable (T=30 a T=500).
- Marco directamente aplicable a datos financieros no estacionarios con cambios de régimen abruptos.
- Entrenamiento exclusivamente sobre datos normales: compatible con el protocolo de no fuga temporal del presente proyecto.

**Limitaciones**

- El vector latente de dimensión fija puede constituir un cuello de botella para series muy largas o de alta dimensionalidad.
- El umbral de detección requiere calibración cuidadosa; la distribución gaussiana del error puede no ser válida en datos financieros con colas pesadas.
- Ausencia de mecanismo de atención: dependencias de largo alcance pueden degradarse para T grande.

**Complejidad computacional**

Arquitectura de dos capas LSTM con 30 unidades por capa (configuración reportada). Parámetros aproximados: $\sim$30K–100K según dimensionalidad de entrada. Tiempo de entrenamiento no reportado explícitamente; escala linealmente con el número de secuencias.

---

### **Modelo 2: OmniAnomaly — Stochastic RNN with VAE**

**Referencia completa**

> Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019). *Robust anomaly detection for multivariate time series through stochastic recurrent neural network*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2828–2837.

**Tipo de arquitectura**

Variational Autoencoder (VAE) integrado con Stochastic Recurrent Neural Network (SRNN). Modelo generativo profundo no supervisado.

**Descripción técnica**

OmniAnomaly modela la distribución latente de series de tiempo multivariadas mediante un VAE cuyo encoder y decoder están parametrizados por RNNs estocásticas. Emplea normalizing flows planares para aproximar distribuciones latentes no-gaussianas, capturando la multimodalidad característica de datos industriales y financieros. La puntuación de anomalía se calcula mediante la probabilidad de reconstrucción (no el MSE escalar), lo que provee un marco probabilístico más riguroso. El umbral se determina mediante el método POT (Peak Over Threshold), basado en la teoría de valores extremos.

**Innovación**

Superación de la asunción gaussiana unimodal de los VAE estándar mediante el flujo normalizador. Tratamiento explícito de la incertidumbre temporal mediante variables estocásticas latentes secuenciales. Mecanismo POT para umbralización que no requiere suposiciones distribucionales sobre el error.

**Tipo de problema**

Detección de anomalías no supervisada en series de tiempo multivariadas con distribuciones complejas y ruido no-gaussiano.

**Datasets utilizados**

SMD (Server Machine Dataset — 28 máquinas), SMAP, MSL (telemetría de naves espaciales NASA).

**Métricas reportadas**

| Dataset | Precision | Recall | F1$_\text{best}$ |
|---------|-----------|--------|-----------------|
| SMD | 0.9260 | 0.9149 | **0.9204** |
| SMAP | 0.9502 | 0.5482 | 0.6953 |
| MSL | 0.9245 | 0.8502 | 0.8858 |

**Fortalezas**

- Marco probabilístico robusto: maneja incertidumbre de forma explícita.
- Desempeño estado-del-arte en SMD (segundo mejor en F1 reportado en la literatura hasta 2022).
- Umbralización POT estadísticamente fundamentada: relevante para distribuciones financieras de cola pesada.

**Limitaciones**

- Ignora información local invariante: ruido de alta frecuencia puede generar falsos positivos.
- Costo computacional elevado debido al flujo normalizador y las variables estocásticas.
- Inferencia variacional aproximada (ELBO) puede subestimar la varianza posterior verdadera.

**Complejidad computacional**

Aproximadamente 200K–500K parámetros según configuración. El flujo normalizador añade overhead computacional significativo respecto a VAE estándar. Tiempo de entrenamiento reportado como mayor que EncDec-AD para dimensiones equivalentes.

---

### **Modelo 3: USAD — UnSupervised Anomaly Detection**

**Referencia completa**

> Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2020). *USAD: UnSupervised Anomaly Detection on Multivariate Time Series*. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

**Tipo de arquitectura**

Dual Autoencoder con entrenamiento adversarial. Aprendizaje no supervisado basado en reconstrucción amplificada.

**Descripción técnica**

USAD entrena dos autoencoders ($AE_1$, $AE_2$) que comparten el mismo encoder pero tienen decoders distintos. El entrenamiento procede en dos fases: (1) ambos AEs aprenden a reconstruir datos normales; (2) $AE_2$ es entrenado adversarialmente para amplificar el error de reconstrucción de $AE_1$ ante anomalías. La puntuación de anomalía combina los errores de reconstrucción de ambos autoencoders mediante un parámetro de balance $\alpha$: $\text{score} = (1-\alpha)\|X - AE_1(X)\| + \alpha\|X - AE_2(AE_1(X))\|$. La entrada al modelo se presenta en orden temporal (ventana deslizante).

**Innovación**

La dualidad adversarial resuelve el problema de compresión excesiva de los AE estándar, que tienden a reconstruir anomalías con error bajo si el espacio latente es suficientemente grande. La amplificación garantiza que las anomalías sean detectables incluso cuando son sutiles.

**Tipo de problema**

Detección de anomalías en series de tiempo multivariadas; especialmente efectivo ante anomalías de baja magnitud.

**Datasets utilizados**

SMD, SMAP, MSL, SWaT (Secure Water Treatment), WADI.

**Métricas reportadas**

| Dataset | Precision | Recall | F1$_\text{best}$ |
|---------|-----------|--------|-----------------|
| SMD | 0.7951 | 0.9418 | 0.8622 |
| SMAP | 0.9032 | 0.8235 | 0.8615 |
| MSL | 0.8684 | 0.9167 | **0.8918** |

**Fortalezas**

- Entrenamiento estable comparado con GANs completos.
- Alta capacidad de recuperación ante anomalías de baja amplitud.
- Recall elevado en SMD y MSL: relevante para detección de cambios de régimen graduales en mercados.

**Limitaciones**

- El parámetro $\alpha$ requiere ajuste; su selección impacta el balance precision/recall.
- Arquitectura de encoder no recurrente en la implementación original: dependencias temporales capturadas solo mediante la ventana de entrada.
- Menor precision que OmniAnomaly en SMD: mayor tasa de falsos positivos.

**Complejidad computacional**

Arquitectura ligera: $\sim$50K–150K parámetros. Tiempo de entrenamiento reducido respecto a modelos VAE con flujos normalizadores. Eficiente para conjuntos de datos de escala moderada.

---

### **Modelo 4: Anomaly Transformer**

**Referencia completa**

> Xu, J., Wu, H., Wang, J., & Long, M. (2022). *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*. International Conference on Learning Representations (ICLR 2022).

**Tipo de arquitectura**

Transformer con mecanismo de atención dual (Prior-Association + Series-Association). Paradigma MINIMAX para amplificación de discrepancias.

**Descripción técnica**

El modelo introduce el concepto de *Association Discrepancy*: para cada timestamp, modela simultáneamente (1) *Prior-Association*, una distribución gaussiana sobre puntos cercanos que aproxima el foco local, y (2) *Series-Association*, los pesos de atención aprendidos sobre toda la serie. La hipótesis central es que los puntos normales establecen asociaciones fuertes con su vecindad local, mientras que las anomalías —al ser raras— no pueden construir patrones de asociación coherentes con la serie completa. La función de pérdida MINIMAX amplifica la discrepancia entre ambas asociaciones, haciendo las anomalías más distinguibles. El error de reconstrucción y la discrepancia de asociación se combinan para la puntuación final.

**Innovación**

Primera propuesta que utiliza el patrón de atención como señal directa de anomalía en lugar de depender exclusivamente del error de reconstrucción. Resuelve la limitación de los Transformers estándar que tienden a reconstruir anomalías con error bajo en espacios de alta dimensión.

**Tipo de problema**

Detección de anomalías en series de tiempo univariadas y multivariadas; modelado de dependencias de largo alcance.

**Datasets utilizados**

SMD, MSL, SMAP, SWaT, PSM (Pooled Server Metrics).

**Métricas reportadas**

Reporta mejoras consistentes sobre baselines previos en todos los datasets benchmark, con F1 superior a TranAD en MSL y SMAP en evaluación point-adjust.

**Fortalezas**

- Captura dependencias de largo alcance mediante self-attention: superior a LSTM para ventanas T > 50.
- La discrepancia de asociación provee interpretabilidad sobre qué timestamps son anómalos.
- Paralelismo computacional: entrenamiento más rápido que modelos LSTM bidireccionales.

**Limitaciones**

- Complejidad cuadrática $O(T^2)$ en la atención: costoso para secuencias largas.
- Requiere positional encoding: puede perder información sobre estructura temporal local en series financieras con alta autocorrelación a lag corto.
- Sensible a la escala del hiperparámetro de temperatura en la distribución gaussiana prior.

**Complejidad computacional**

Aproximadamente 2M–8M parámetros según profundidad de capas. La complejidad cuadrática en la longitud de secuencia T limita su aplicabilidad directa a T > 200 sin modificaciones.

---

### **Modelo 5: TranAD — Deep Transformer Networks for Anomaly Detection**

**Referencia completa**

> Tuli, S., Casale, G., & Jennings, N. R. (2022). *TranAD: Deep transformer networks for anomaly detection in multivariate time series data*. Proceedings of the VLDB Endowment, 15, 1201–1214.

**Tipo de arquitectura**

Transformer encoder-decoder con self-conditioning y entrenamiento adversarial. Paradigma híbrido reconstrucción + adversarial.

**Descripción técnica**

TranAD incorpora dos innovaciones sobre el Transformer estándar para detección de anomalías: (1) *Self-conditioning*: el decoder recibe como entrada adicional una representación intermedia del encoder, mejorando la estabilidad del gradiente y la recuperación de características ante anomalías sutiles; (2) *Adversarial training*: el modelo es entrenado de forma que el error de reconstrucción sea amplificado para anomalías, similar a USAD pero sobre arquitectura transformer. La puntuación de anomalía considera el error de reconstrucción de ambas salidas del decoder.

**Innovación**

Combina la eficiencia computacional de los Transformers con la amplificación adversarial de USAD. El self-conditioning garantiza robustez en la extracción de características incluso cuando los datos contienen instancias ruidosas o contaminadas durante el entrenamiento.

**Tipo de problema**

Detección de anomalías en series de tiempo multivariadas a gran escala; aplicable en sistemas con alta dimensionalidad.

**Datasets utilizados**

SMD, MSL, SMAP, SWaT, WADI, MBA (Mars Science Laboratory y SMAP benchmark).

**Métricas reportadas**

Reporta F1 competitivo con Anomaly Transformer en múltiples benchmarks; ventaja específica en datasets de alta dimensionalidad (WADI, SWaT) donde los modelos LSTM enfrentan limitaciones de escalabilidad.

**Fortalezas**

- Eficiencia en entrenamiento y evaluación respecto a OmniAnomaly.
- Estabilidad del entrenamiento adversarial gracias al self-conditioning.
- Escalabilidad a alta dimensionalidad: superior a LSTM para $d > 20$ variables.

**Limitaciones**

- Complejidad arquitectónica elevada: dificulta la interpretación de los componentes de error.
- La ventaja sobre EncDec-AD no está claramente demostrada en contextos univariados o de baja dimensionalidad (como el presente proyecto con $d=3$).
- Requiere dataset de tamaño considerable para que el entrenamiento adversarial converja establemente.

**Complejidad computacional**

Aproximadamente 1M–5M parámetros. Tiempo de entrenamiento reportado como significativamente menor que OmniAnomaly para datasets equivalentes.

---

### **Modelo 6: DeepAnT — Deep Learning for Anomaly Detection in Time Series**

**Referencia completa**

> Munir, M., Siddiqui, S. A., Dengel, A., & Ahmed, S. (2019). *DeepAnT: A deep learning approach for unsupervised anomaly detection in time series*. IEEE Access, 7, 1991–2005.

**Tipo de arquitectura**

CNN predictor (forecasting-based anomaly detection). No supervisado.

**Descripción técnica**

DeepAnT utiliza una CNN de dos capas de convolución temporal seguida de capas densas para predecir el siguiente valor (o ventana de valores) en la serie de tiempo. La detección de anomalías se realiza comparando el valor predicho con el observado: diferencias superiores a un umbral adaptativo constituyen anomalías. La red aprende patrones locales de la serie normal mediante convoluciones unidimensionales con kernels de tamaño variable. Es aplicable tanto a series univariadas como multivariadas.

**Innovación**

Primer uso sistemático de CNN (en lugar de LSTM) para anomaly detection en series de tiempo, demostrando que la captura de patrones locales mediante convolución es competitiva con modelos recurrentes para series con estructura temporal de corto alcance. Manejo de contaminación en datos de entrenamiento inferior al 5%.

**Tipo de problema**

Detección de anomalías puntuales, contextuales y discordancias en series de tiempo univariadas y multivariadas.

**Datasets utilizados**

Datasets de NASA (SMAP, MSL), Yahoo Anomaly Benchmark, datasets sintéticos con anomalías inducidas.

**Métricas reportadas**

Precision y Recall superiores al 85% en Yahoo Benchmark (S5); desempeño competitivo ante métodos clásicos (LOF, Isolation Forest) y LSTM-AD en datasets predecibles.

**Fortalezas**

- Eficiencia computacional: entrenamiento y evaluación significativamente más rápidos que modelos LSTM.
- Robustez ante contaminación de datos de entrenamiento inferior al 5%.
- Aplicable a series con patrones locales dominantes (ciclos de corto periodo).

**Limitaciones**

- Basado en predicción: ineficaz ante series impredecibles (como retornos financieros a corto plazo).
- Las CNNs no capturan dependencias de largo alcance de forma nativa: limitación crítica para cambios de régimen graduales.
- El umbral adaptativo no dispone de fundamentación estadística robusta en colas pesadas.

**Complejidad computacional**

Arquitectura ligera: $\sim$10K–50K parámetros. Tiempo de entrenamiento mínimo; adecuado para aplicaciones en tiempo real o con recursos computacionales limitados.

---

### **Modelo 7: CAE-AD — Contrastive Autoencoder for Anomaly Detection**

**Referencia completa**

> Zhou, H., Yu, K., Zhang, X., Wu, G., & Yazidi, A. (2022). *Contrastive autoencoder for anomaly detection in multivariate time series*. Information Sciences, 610, 266–280.

**Tipo de arquitectura**

LSTM Autoencoder con aprendizaje contrastivo. Paradigma de representación invariante local.

**Descripción técnica**

CAE-AD extiende el EncDec-AD (LSTM encoder-decoder) incorporando un módulo de aprendizaje contrastivo sobre el espacio latente. El objetivo contrastivo minimiza la distancia entre representaciones latentes de instancias similares (ventanas del mismo régimen) y maximiza la distancia entre instancias disímiles. El ruido gaussiano se utiliza como forma de data augmentation para generar pares positivos. La función de pérdida combina el error de reconstrucción estándar con la pérdida contrastiva InfoNCE. El encoder LSTM tiene dimensión oculta 64 y dimensión latente $z = 18$; la ventana de entrada es $w=36$ con stride $l=10$.

**Innovación**

Introduce la propiedad de invarianza local en el espacio latente de un LSTM AE: el modelo aprende representaciones que son robustas a pequeñas perturbaciones del régimen normal, reduciendo los falsos positivos causados por ruido de alta frecuencia que afectan a OmniAnomaly. Primer uso de contrastive learning sobre LSTM AE para TSAD.

**Tipo de problema**

Detección de anomalías en series de tiempo multivariadas con ruido; particularmente efectivo en datasets con alta proporción de ruido en datos de entrenamiento.

**Datasets utilizados**

SMD (28 máquinas), SMAP (55 canales), MSL (27 canales).

**Métricas reportadas**

| Dataset | Precision | Recall | F1$_\text{best}$ |
|---------|-----------|--------|-----------------|
| SMD | 0.9265 | 0.9491 | **0.9376** |
| SMAP | 0.8824 | 0.9836 | **0.9302** |
| MSL | 0.8611 | 0.9688 | **0.9119** |

Mejores F1$_\text{best}$ reportados en los tres benchmarks en el momento de publicación.

**Fortalezas**

- Estado del arte en SMD, SMAP y MSL al momento de publicación.
- Robustez ante ruido: el aprendizaje contrastivo previene que el modelo asigne scores elevados a variaciones normales de alta frecuencia.
- Arquitectura LSTM: captura dependencias temporales de corto y mediano alcance sin la complejidad cuadrática del Transformer.

**Limitaciones**

- Requiere hiperparámetros adicionales (temperatura $\sigma$, desviación del ruido $r_1, r_2$) cuya calibración impacta significativamente el desempeño.
- El data augmentation por ruido gaussiano puede ser inadecuado si la distribución de ruido en los datos reales es no-gaussiana (e.g., distribuciones de cola pesada en retornos financieros).
- La ventana fija $w=36$ puede no ser óptima para todos los activos; requiere validación por activo.

**Complejidad computacional**

Encoder LSTM con dimensión oculta 64: $\sim$100K–200K parámetros según número de features de entrada. La pérdida contrastiva añade overhead computacional por el cálculo de similitudes en el batch; el tiempo de entrenamiento es superior a EncDec-AD estándar en un factor aproximado de 1.3–1.8x.

---

## 3.4 Comparación Crítica

### **Tabla Comparativa Estructurada**

| Modelo | Arquitectura | Paradigma | F1 SMD | F1 SMAP | F1 MSL | Parámetros | Escalabilidad | Interpretabilidad |
|--------|-------------|-----------|--------|---------|--------|-----------|--------------|------------------|
| EncDec-AD | LSTM Enc-Dec | Reconstrucción | 0.7729 | 0.8688 | 0.8307 | ~50K | Alta | Media |
| OmniAnomaly | VAE+SRNN | Generativo | **0.9204** | 0.6953 | 0.8858 | ~300K | Media | Baja |
| USAD | Dual AE | Adversarial | 0.8622 | 0.8615 | 0.8918 | ~100K | Alta | Media |
| Anomaly Transformer | Transformer | Atención dual | — | — | — | ~5M | Baja (O(T²)) | Alta |
| TranAD | Transformer Adv. | Adversarial | — | — | — | ~3M | Media | Media |
| DeepAnT | CNN | Predicción | — | — | — | ~30K | Muy Alta | Baja |
| CAE-AD | LSTM Contrastivo | Reconstrucción+Contraste | **0.9376** | **0.9302** | **0.9119** | ~150K | Alta | Media |

*Nota: Las celdas vacías corresponden a modelos que no reportan resultados en los benchmarks SMD/SMAP/MSL estándar bajo el mismo protocolo de evaluación.*

---

### **Análisis de Tendencias**

**Tendencia 1: Transición de modelos predictivos a reconstrucción**

La literatura muestra una migración clara desde modelos basados en predicción hacia modelos de reconstrucción. Los modelos predictivos (LSTM-AD, DeepAnT) asumen que la serie es predecible, supuesto inválido en retornos financieros a corto plazo. Los modelos de reconstrucción (EncDec-AD, OmniAnomaly, CAE-AD) aprenden el patrón de normalidad sin requerir predictibilidad, lo que los hace intrínsecamente más adecuados para series financieras.

**Tendencia 2: Irrupción de los Transformers (2022–presente)**

A partir de 2022, los modelos basados en Transformer (Anomaly Transformer, TranAD) reportan resultados competitivos o superiores a los modelos LSTM en datasets de alta dimensionalidad. Sin embargo, esta ventaja se atenúa considerablemente en configuraciones de baja dimensionalidad ($d \leq 5$) y secuencias cortas ($T \leq 50$), donde la complejidad cuadrática no se amortiza y los mecanismos de atención no logran diferenciarse del modelado recurrente estándar.

**Tendencia 3: Modelos híbridos y aprendizaje contrastivo**

La incorporación de objetivos auxiliares (aprendizaje contrastivo en CAE-AD, discrepancia de asociación en Anomaly Transformer, adversarial training en USAD/TranAD) representa la tendencia más reciente y consistente en mejorar el F1 sobre arquitecturas base. El aprendizaje contrastivo aplicado sobre espacios latentes LSTM (CAE-AD) logra el mejor desempeño en los tres benchmarks principales al momento de publicación, con una fracción del costo computacional de los Transformers profundos.

---

### **Identificación del Gap en la Literatura**

A pesar del progreso documentado, se identifican tres brechas relevantes para el contexto del presente proyecto:

**Gap 1: Ausencia de validación en mercados emergentes.**
Los benchmarks dominantes (SMD, SMAP, MSL) corresponden a datos industriales y de telemetría. La generalización a series financieras de mercados emergentes latinoamericanos —caracterizadas por menor liquidez, mayor clustering de volatilidad, y regímenes de crisis más abruptos (e.g., COVID-19, choques del precio del petróleo)— no ha sido sistemáticamente validada.

**Gap 2: Tratamiento del problema de umbralización en distribuciones de cola pesada.**
La mayoría de los modelos revisados utilizan umbrales basados en percentiles empíricos o distribuciones gaussianas del error de reconstrucción. Los retornos financieros presentan colas pesadas (exceso de curtosis documentado en el Notebook 1 del presente proyecto), lo que implica que los umbrales basados en normalidad subestimarán la frecuencia de eventos extremos legítimos y sobreestimarán la tasa de falsos positivos.

**Gap 3: Modelos de activo único vs. portafolio.**
La literatura se centra predominantemente en detección multivariada de alta dimensionalidad o en detección univariada aislada. El presente proyecto opera en un espacio intermedio: activos colombianos (EC, CIB, AVAL, TGLS) con correlaciones conocidas entre sí, para los cuales un modelo por activo con feature vector de baja dimensionalidad ($d=3$) puede capturar los regímenes locales de cada instrumento sin la complejidad computacional de los modelos multivariados de alta dimensión.

---

### **Justificación del Modelo Seleccionado para el Presente Proyecto**

En función del análisis crítico comparativo, el presente proyecto propone un **Denoising Autoencoder recurrente con dos variantes arquitectónicas: DAE-LSTM y DAE-GRU**. Ambas variantes siguen el paradigma de reconstrucción no supervisada de EncDec-AD (Malhotra et al., 2016) y compiten entre sí bajo el mismo protocolo experimental. La variante con mejor desempeño en el conjunto de validación (período COVID-2020) se reporta como modelo final del proyecto.

La justificación de esta elección se fundamenta en los siguientes criterios:

- **Alineación con el paradigma no supervisado**: el entrenamiento exclusivo sobre datos normales (2015–2019) es compatible con la ausencia de etiquetas de anomalía verificadas para los ADRs colombianos.
- **Robustez ante impredecibilidad de retornos**: los log-retornos financieros son inherentemente impredecibles a corto plazo; el paradigma de reconstrucción no requiere predictibilidad de la serie, a diferencia de los modelos forecasting-based (DeepAnT, LSTM-AD).
- **Baja dimensionalidad del feature vector** ($d=3$, $T=30$): los Transformers (Anomaly Transformer, TranAD) no presentan ventaja computacional ni de desempeño documentada en esta escala; su complejidad cuadrática $O(T^2)$ no se amortiza para $T=30$.
- **Comparación DAE-LSTM vs. DAE-GRU**: la competencia entre variantes permite aislar el efecto de la arquitectura de celda recurrente (compuertas LSTM vs. GRU) sobre la capacidad de detección, manteniendo constantes todos los demás factores del pipeline.
- **Umbral percentil empírico**: consistente con la evidencia de colas pesadas identificada en el EDA y con la recomendación de no asumir distribución gaussiana del error de reconstrucción.

El benchmark experimental incluye cinco comparadores metodológicamente válidos para el marco no supervisado: Z-Score estadístico, Isolation Forest, One-Class SVM, LSTM Predictor y GRU Predictor. Los modelos supervisados (Random Forest, XGBoost) fueron **excluidos** por requerir etiquetas de anomalía durante el entrenamiento, lo que es incompatible con el marco no supervisado del proyecto y constituiría una comparación metodológicamente injusta.

Los modelos Transformer (Anomaly Transformer, TranAD) y el modelo contrastivo (CAE-AD) quedan identificados como extensiones naturales del trabajo futuro, condicionadas a la disponibilidad de datos de mayor dimensionalidad o a la extensión del análisis a portafolios multivariados de los cuatro activos colombianos.