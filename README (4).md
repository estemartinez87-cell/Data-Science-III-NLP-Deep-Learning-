# Clasificador Supervisado con TF-IDF — AG News

Checkpoint del Módulo 3. Clasificación de noticias en 4 categorías (**Business**, **Sci_Tech**, **Sports**, **World**) usando vectorización TF-IDF y modelos de aprendizaje supervisado clásicos. Reutiliza el pipeline de preprocesamiento (limpieza con Regex + lematización con SpaCy) desarrollado en el Módulo 2.

> ⚠️ **Nota**: esta versión del README corresponde al pipeline actualizado, que agrega una etapa de experimentación real de vectorización (ver sección 2). Los resultados numéricos de las tablas de abajo están marcados como `TODO` porque este script todavía no se corrió — hay que reemplazarlos por la salida real una vez ejecutado el notebook completo.

## Dataset

| | Ejemplos | Clases |
|---|---|---|
| Train | 8,000 | Business, Sci_Tech, Sports, World |
| Test | 2,000 | Business, Sci_Tech, Sports, World |

Dataset balanceado: 500 ejemplos de test por clase.

## Pipeline

1. **Carga de datos**: desde Google Drive (mismos splits del Módulo 2).
2. **Limpieza de texto**: decodificación HTML, eliminación de URLs y etiquetas, normalización de caracteres especiales.
3. **Lematización**: procesamiento por lotes con SpaCy (`en_core_web_sm`), eliminando puntuación, espacios y stopwords.
4. **Experimentación de vectorización TF-IDF** (sin data leakage: `fit_transform` solo en train, `transform` en test), comparando 4 configuraciones con Logistic Regression fijo como clasificador de referencia:

   | Configuración | max_features | ngram_range |
   |---|---|---|
   | Unigrama, sin límite de vocab | `None` | `(1, 1)` |
   | Unigrama, max_features=5000 | `5000` | `(1, 1)` |
   | Uni+bigrama, sin límite de vocab | `None` | `(1, 2)` |
   | Uni+bigrama, max_features=5000 | `5000` | `(1, 2)` |

   La configuración con mejor F1-macro se selecciona como `best_cfg` y se reutiliza en el paso siguiente.
5. **Comparación de modelos** sobre `best_cfg`: Multinomial Naive Bayes, Logistic Regression, Linear SVM.

## Resultados — Experimentación de vectorización

| Configuración | Tamaño de vocabulario | Accuracy | F1-macro |
|---|---|---|---|
| Unigrama, sin límite de vocab | TODO | TODO | TODO |
| Unigrama, max_features=5000 | TODO | TODO | TODO |
| Uni+bigrama, sin límite de vocab | TODO | TODO | TODO |
| Uni+bigrama, max_features=5000 | TODO | TODO | TODO |

**Mejor configuración (`best_cfg`):** TODO

## Resultados — Comparación de modelos (con `best_cfg`)

| Modelo | Accuracy | F1-macro |
|---|---|---|
| Naive Bayes | TODO | TODO |
| Logistic Regression | TODO | TODO |
| Linear SVM | TODO | TODO |

**Modelo elegido:** TODO

### Reporte de clasificación (modelo ganador)

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Business | TODO | TODO | TODO | 500 |
| Sci_Tech | TODO | TODO | TODO | 500 |
| Sports | TODO | TODO | TODO | 500 |
| World | TODO | TODO | TODO | 500 |
| **Accuracy** | | | TODO | 2000 |

## Análisis de errores

Total de instancias mal clasificadas: **TODO / 2000**.

Top confusiones (real → predicho : cantidad):

| Real | Predicho | Cantidad |
|---|---|---|
| TODO | TODO | TODO |

> Referencia: en la versión anterior (con `best_cfg` fijo en `max_features=10000, ngram_range=(1,2)`), Naive Bayes había dado 90.25% de accuracy, con las confusiones más frecuentes concentradas en Business↔Sci_Tech y World↔Business. Es esperable que estos números cambien (para mejor o peor) al usar el `best_cfg` encontrado por experimentación real.

## Conclusiones

*(Sección a completar con los resultados reales. Guía de qué analizar, en base a lo que ya sabemos del dominio):*

1. **Comparar el `best_cfg` encontrado contra el valor fijo anterior** (`max_features=10000, ngram_range=(1,2)`): ¿la experimentación confirma que ese rango era razonable, o el vocabulario sin límite / con bigramas cambia sustancialmente el resultado? Esto valida (o no) la intuición original.
2. **Relación entre tamaño de vocabulario y F1-macro**: si las configuraciones sin límite de `max_features` no superan (o apenas superan) a las de `max_features=5000`, es evidencia de que el poder discriminativo del vocabulario se satura rápido — relevante para justificar una configuración más liviana en producción.
3. **Persistencia del patrón de confusión Business↔Sci_Tech↔World**: si el nuevo modelo mantiene los mismos pares de error que la versión anterior, refuerza la hipótesis de que el problema es de ambigüedad semántica del dominio (noticias que legítimamente combinan economía, tecnología y geopolítica) y no un artefacto de la configuración de vectorización — es decir, no se resuelve solo ajustando `max_features` o `ngram_range`.
4. **Costo computacional de la experimentación**: vale la pena registrar cuánto tardó la etapa de comparación de las 4 configuraciones vs. el tiempo total del pipeline, para justificar si conviene incorporar esta búsqueda como paso estándar o si el retorno no justifica el costo frente a fijar un valor razonable de entrada.
5. **Conclusión general**: si el `best_cfg` encontrado da métricas similares (dentro de un margen pequeño) a las del valor fijo anterior, la conclusión práctica es que la elección original ya era casi óptima y la experimentación sirve como validación más que como mejora sustancial. Si en cambio hay una diferencia notable, documentar cuál configuración ganó y por qué (unigramas vs bigramas, límite de vocabulario) para dejarlo como decisión justificada y no arbitraria.

## Archivos generados

- `classification_report.txt` — reporte completo de clasificación del modelo ganador.
- `confusion_matrix.png` — matriz de confusión visual del modelo ganador (con `best_cfg` en el título).
- `summary.json` — resumen estructurado: incluye ahora `vec_experiments` (detalle de las 4 configuraciones de TF-IDF probadas), además de la comparación de modelos y las top confusiones.
