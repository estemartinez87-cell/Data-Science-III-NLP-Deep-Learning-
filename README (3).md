# Clasificador Supervisado con TF-IDF — AG News

Checkpoint del Módulo 3. Clasificación de noticias en 4 categorías (**Business**, **Sci_Tech**, **Sports**, **World**) usando vectorización TF-IDF y modelos de aprendizaje supervisado clásicos. Reutiliza el pipeline de preprocesamiento (limpieza con Regex + lematización con SpaCy) desarrollado en el Módulo 2.

## Dataset

| | Ejemplos | Clases |
|---|---|---|
| Train | 8,000 | Business, Sci_Tech, Sports, World |
| Test | 2,000 | Business, Sci_Tech, Sports, World |

Dataset balanceado: 500 ejemplos de test por clase.

## Pipeline

1. **Limpieza de texto**: decodificación HTML, eliminación de URLs y etiquetas, normalización de caracteres especiales.
2. **Lematización**: procesamiento por lotes con SpaCy (`en_core_web_sm`), eliminando puntuación, espacios y stopwords.
3. **Vectorización TF-IDF**: `max_features=10000`, `ngram_range=(1,2)` (unigramas + bigramas).
4. **Modelos evaluados**: Multinomial Naive Bayes, Logistic Regression, Linear SVM.

## Resultados — Comparación de modelos

| Modelo | Accuracy | F1-macro |
|---|---|---|
| **Naive Bayes** | **0.9025** | **0.9025** |
| Logistic Regression | 0.8980 | 0.8979 |
| Linear SVM | 0.8940 | 0.8940 |

**Modelo elegido:** Naive Bayes.

### Reporte de clasificación (Naive Bayes)

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Business | 0.8524 | 0.8660 | 0.8591 | 500 |
| Sci_Tech | 0.8853 | 0.8800 | 0.8826 | 500 |
| Sports | 0.9547 | 0.9700 | 0.9623 | 500 |
| World | 0.9179 | 0.8940 | 0.9058 | 500 |
| **Accuracy** | | | **0.9025** | 2000 |

## Análisis de errores

Total de instancias mal clasificadas: **195 / 2000 (9.75%)**.

Top confusiones (real → predicho : cantidad):

| Real | Predicho | Cantidad |
|---|---|---|
| Sci_Tech | Business | 41 |
| Business | Sci_Tech | 40 |
| World | Business | 30 |
| Business | World | 23 |
| World | Sports | 14 |
| Sci_Tech | World | 14 |
| World | Sci_Tech | 9 |
| Sports | Sci_Tech | 8 |

## Conclusiones

**1. Naive Bayes fue el mejor baseline, y no por poco margen conceptual.**
Con textos cortos y muy dispersos como titulares de noticias, la independencia condicional que asume Naive Bayes deja de ser una simplificación tan costosa: cada término aporta evidencia casi aislada sobre la clase, que es justo el escenario donde este modelo suele destacarse frente a métodos más expresivos como SVM o regresión logística, que necesitan más señal para estimar bien sus pesos. La diferencia con los otros dos modelos (0.90 vs 0.898 vs 0.894 de F1-macro) es pequeña pero consistente, y sugiere que el techo de rendimiento con esta representación (TF-IDF + n-gramas 1-2) ya está bastante cerca para los tres algoritmos: la elección del modelo aporta menos que la elección de las features.

**2. El error se concentra en confusiones semánticamente explicables, no en fallos aleatorios.**
El 74% de los errores más frecuentes (41+40+30+23 de 195) involucra a la clase **Business** cruzándose con **Sci_Tech** y **World**. Esto no es ruido del modelo: refleja ambigüedad real en el dominio periodístico. Noticias sobre grandes empresas tecnológicas (ej. resultados financieros de una compañía de tecnología) comparten vocabulario económico y técnico simultáneamente, y noticias geopolíticas con impacto económico (acuerdos comerciales, sanciones, negociaciones internacionales) legítimamente pertenecen a la frontera entre World y Business. El ejemplo de "EU, US Talks on Aircraft Aid" clasificado como World cuando la etiqueta real es Business ilustra exactamente este solapamiento.

**3. Sports es la clase mejor resuelta, y por una razón estructural.**
Con F1=0.9623, Sports tiene el vocabulario más distintivo y menos solapado del dataset (nombres propios de deportistas, equipos, resultados numéricos, terminología deportiva específica). Es consistente con que TF-IDF —que premia términos discriminativos poco frecuentes en el corpus global— funcione especialmente bien cuando el vocabulario de una clase es léxicamente autocontenido.

**4. Las confusiones World↔Sports (14 casos) apuntan a una limitación real del enfoque bag-of-words.**
El ejemplo "Australia 362-7 v India, third test" etiquetado como World pero predicho como Sports muestra que el criterio de etiquetado original del dataset no siempre coincide con el contenido léxico superficial: una noticia de cricket entre selecciones nacionales fue clasificada como World por su marco geopolítico/internacional, pero su vocabulario (resultados, "test", nombres de países en contexto deportivo) es indistinguible del de Sports para un modelo que solo ve términos. Esto sugiere que una parte del error observado no es corregible únicamente ajustando el modelo o el vectorizador — es un límite del criterio de etiquetado frente a una representación puramente léxica.

**5. Rendimiento global (90.25% accuracy) es sólido para un baseline TF-IDF sin ajuste fino de hiperparámetros.**
Considerando que no hubo búsqueda exhaustiva de la configuración óptima del vectorizador (max_features y ngram_range quedaron fijos en `(10000, (1,2))`), el resultado es un piso razonable. Los próximos pasos con mayor potencial de mejora, en orden de esfuerzo/impacto, serían:
- Recuperar y ejecutar una búsqueda real de hiperparámetros de TF-IDF (grid search sobre `max_features` y `ngram_range`) en lugar del valor fijo actual.
- Probar `class_weight="balanced"` en Logistic Regression y Linear SVM para ver si mejora específicamente el par Business/Sci_Tech.
- Incorporar features adicionales (longitud del texto, entidades nombradas de SpaCy) que podrían ayudar a desambiguar Business vs. World, ya que el conflicto entre estas dos clases no parece resolverse con más vocabulario sino con contexto estructural.
- Evaluar un modelo con embeddings preentrenados (o TF-IDF + reducción de dimensionalidad con SVD) para capturar relaciones semánticas que el bag-of-words no puede representar, especialmente para las confusiones World↔Sports que parecen tener origen en el criterio de etiquetado más que en el vocabulario.

## Archivos generados

- `classification_report.txt` — reporte completo de clasificación del modelo ganador.
- `confusion_matrix.png` — matriz de confusión visual (Naive Bayes).
- `summary.json` — resumen estructurado (configuración, métricas por modelo, top confusiones).
