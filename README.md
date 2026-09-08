# Clasificación de Noticias — TF-IDF Clásico vs. Transformer + LoRA

Proyecto de NLP sobre AG News (4 clases: **Business**, **Sci_Tech**, **Sports**, **World**), que compara un pipeline clásico de vectorización TF-IDF con un fine-tuning eficiente (PEFT/LoRA) de un Transformer preentrenado.

## Dataset

| | Ejemplos | Clases |
|---|---|---|
| Train | 8,000 | Business, Sci_Tech, Sports, World |
| Test | 2,000 | Business, Sci_Tech, Sports, World |

Dataset balanceado: 500 ejemplos de test por clase.

## Parte A — Baseline: TF-IDF + clasificadores clásicos

**Pipeline**: limpieza Regex + lematización SpaCy → experimentación de vectorización TF-IDF (4 configuraciones) → comparación de 3 modelos clásicos.

### Experimentación de vectorización

| Configuración | Vocabulario | Accuracy | F1-macro |
|---|---|---|---|
| **Unigrama, sin límite de vocab** | **17,513** | **0.8970** | **0.8969** |
| Unigrama, max_features=5000 | 5,000 | 0.8945 | 0.8944 |
| Uni+bigrama, sin límite de vocab | 155,044 | 0.8965 | 0.8963 |
| Uni+bigrama, max_features=5000 | 5,000 | 0.8935 | 0.8935 |

**Mejor configuración**: unigrama sin límite de vocabulario (F1-macro=0.8969). Interesante: agregar bigramas o limitar el vocabulario no mejora el resultado — el poder discriminativo ya está en las palabras individuales.

### Comparación de modelos (con la mejor configuración)

| Modelo | Accuracy | F1-macro |
|---|---|---|
| **Naive Bayes** | **0.8970** | **0.8973** |
| Logistic Regression | 0.8970 | 0.8969 |
| Linear SVM | 0.8920 | 0.8919 |

**Modelo elegido**: Naive Bayes.

### Reporte de clasificación (Naive Bayes)

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Business | 0.8243 | 0.8820 | 0.8522 |
| Sci_Tech | 0.8947 | 0.8500 | 0.8718 |
| Sports | 0.9639 | 0.9600 | 0.9619 |
| World | 0.9106 | 0.8960 | 0.9032 |
| **Accuracy** | | | **0.8970** |

## Parte B — Fine-tuning eficiente con LoRA (PEFT)

**Modelo base**: `distilbert-base-uncased`, elegido por idioma (corpus en inglés) y eficiencia de cómputo (66M parámetros vs. 110M de BERT-base, ~97% de su rendimiento).

**Configuración de LoRA**:
| Hiperparámetro | Valor |
|---|---|
| r (rango) | 8 |
| lora_alpha | 16 |
| target_modules | `q_lin`, `v_lin` |
| lora_dropout | 0.1 |
| bias | none |

**Parámetros**: 741,124 entrenables de 67,697,672 totales (**1.0948%**) — cumple el requisito de <3%.

**Entrenamiento**: 3 épocas, learning rate 2e-5, batch size 16. **Tiempo: 188.61 minutos** (corrido sin GPU — con GPU real el mismo entrenamiento toma entre 5 y 15 minutos).

**Pérdidas**: train (última, step logging) 0.3296 | train (promedio general) 0.4736 | validación 0.3166.

### Reporte de clasificación (DistilBERT + LoRA)

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Business | 0.8236 | 0.8500 | 0.8366 |
| Sci_Tech | 0.8643 | 0.8660 | 0.8651 |
| Sports | 0.9682 | 0.9740 | 0.9711 |
| World | 0.9187 | 0.8820 | 0.9000 |
| **Accuracy** | | | **0.8930** |

## Benchmark final: Baseline vs. Transformer + LoRA

| Métrica | Naive Bayes + TF-IDF | DistilBERT + LoRA | Delta |
|---|---|---|---|
| Accuracy | 0.8970 | 0.8930 | -0.0040 |
| Precision (macro) | 0.8984 | 0.8937 | -0.0047 |
| Recall (macro) | 0.8970 | 0.8930 | -0.0040 |
| **F1 (macro)** | **0.8973** | **0.8932** | **-0.0041** |

## Conclusión técnica

El Transformer + LoRA **no superó** al baseline clásico — quedó 0.41 puntos de F1-macro por debajo, y tardó 188.61 minutos frente a los pocos segundos que necesita TF-IDF + Naive Bayes. **El incremento de rendimiento no justifica el costo computacional adicional** en este caso: para un dataset de tamaño moderado (8,000 ejemplos) con clases léxicamente bien diferenciadas como AG News, TF-IDF ya captura casi toda la señal discriminativa disponible, dejando poco margen para que un Transformer preentrenado lo supere sin más datos, más épocas, o hiperparámetros de LoRA más ajustados.

Ambos modelos comparten el mismo patrón de dificultad por clase: **Sports** es la mejor resuelta (vocabulario muy distintivo) y **Business** la más difícil (se confunde semánticamente con Sci_Tech y World), lo que confirma que la limitación no es de vectorización sino del dominio: hay ambigüedad real entre noticias económicas, tecnológicas y geopolíticas.

## Archivos generados

- `classification_report_baseline.txt` / `classification_report_lora.txt`
- `confusion_matrix_baseline.png` / `confusion_matrix_comparison.png`
- `benchmark_comparison.csv` / `benchmark_comparison.png`
- `summary_lora.json` — resumen estructurado completo (baseline + LoRA)
- `Checkpoint_NLP3_LoRA.pdf` — reporte técnico formal (estructura: Resumen de arquitectura, Configuración PEFT, Resultados y Comparativa, Conclusiones)
