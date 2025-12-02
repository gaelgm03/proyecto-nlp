# Clasificación de Sentimientos y Quejas en Encuestas de Satisfacción

## Descripción
Proyecto de NLP para dos tareas de clasificación en comentarios de encuestas de satisfacción:
1. **Clasificación de Sentimiento**: positivo, neutro, negativo
2. **Clasificación de Tipo de Queja**: precio, alimentos, infraestructura, ninguna

Compara modelos tradicionales de ML con redes neuronales para identificar el mejor modelo de cada tipo.

## Dataset
- **Fuente**: Encuestas de satisfacción de cafetería universitaria (sintéticas generadas con GPT)
- **Tamaño**: ~1000 registros
- **Columnas**: `role`, `genre`, `age`, `comment`, `kind_of_comment`, `complaint`
- **Clases de Sentimiento**: positivo, neutro, negativo
- **Clases de Queja**: precio, sabor/variedad (alimentos), instalaciones, ninguna

## Modelos Implementados

### Tradicionales (scikit-learn)
| Modelo | Descripción |
|--------|-------------|
| **Logistic Regression** | Clasificación lineal con regularización |
| **SVM (LinearSVC)** | Support Vector Machine lineal |
| **Random Forest** | Ensemble de árboles de decisión |

### Redes Neuronales (TensorFlow/Keras)
| Modelo | Descripción |
|--------|-------------|
| **FNN** | Feedforward Neural Network con Embeddings |
| **CNN** | Red Convolucional 1D para secuencias |
| **LSTM** | Long Short-Term Memory (RNN) |

## Estructura del Proyecto
```
proyecto-nlp/
├── data/                          # Datasets
│   └── respuestas_cafeteria.csv
├── notebooks/                     # Jupyter notebooks
│   └── 01_model_comparison.ipynb  # Comparación completa de modelos
├── src/                           # Código fuente
│   ├── preprocessing.py           # Preprocesamiento, vectorización, etiquetado
│   ├── traditional/               # Modelos tradicionales
│   │   └── models.py
│   └── neural/                    # Redes neuronales
│       ├── models.py              # Clasificador de sentimiento (texto)
│       └── complaint_models.py    # Clasificador de quejas (embeddings spaCy)
├── models/                        # Modelos entrenados
├── respuestas_cafeteria.csv       # Dataset principal
├── CNN_Proyecto (1).ipynb         # Notebook original CNN quejas (referencia)
├── Create_DB.ipynb                # Generación de datos sintéticos
├── Kind_Classifier.ipynb          # Experimentos iniciales (SVM)
├── requirements.txt               # Dependencias
└── README.md
```

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows)
.venv\Scripts\activate

# Activar (Linux/Mac)
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo de spaCy para español
python -m spacy download es_core_news_md
```

## Uso Rápido

### 1. Entrenar y comparar todos los modelos
```python
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_dataframe, vectorize_dataframe, prepare_complaint_labels
from src.traditional.models import train_and_compare_models
from src.neural.models import train_and_compare_neural_models
from src.neural.complaint_models import train_and_compare_complaint_models

# Cargar y preprocesar datos
df = pd.read_csv('respuestas_cafeteria.csv')
df = preprocess_dataframe(df, text_column='comment', output_column='clean_comment')
df = vectorize_dataframe(df, text_column='comment', output_column='vector')
df, complaint_names = prepare_complaint_labels(df, complaint_column='complaint')

# Variables
X_text = df['clean_comment']
X_vectors = np.vstack(df['vector'].values)
y_sentiment = df['kind_of_comment']
y_complaint = df['complaint_category']

# SENTIMIENTO → Modelos Tradicionales (3 modelos)
sent_trad, _ = train_and_compare_models(X_text, y_sentiment)

# TIPO DE QUEJA → Redes Neuronales (3 modelos)
comp_neural, _ = train_and_compare_complaint_models(X_vectors, y_complaint.values, epochs=100)
```

### 2. Predicción con modelos pre-entrenados
```python
from src.predict import CafeteriaAnalyzer

# Cargar analizador (requiere modelos ya entrenados)
analyzer = CafeteriaAnalyzer(
    sentiment_model_path='models/sentiment_best_random_forest.joblib',
    complaint_model_path='models/complaint_best_fnn.h5'
)

# Analizar un comentario
resultado = analyzer.analyze("La comida está muy cara y sin sabor")
print(resultado['sentimiento']['prediccion'])  # 'negativo'
print(resultado['tipo_queja']['prediccion'])   # 'Por precio'
```

> **Nota**: Los modelos se entrenan ejecutando `notebooks/01_model_comparison.ipynb`

## Pipeline de Preprocesamiento
1. Convertir a minúsculas
2. Eliminar caracteres especiales (mantener acentos)
3. Tokenización (NLTK)
4. Eliminación de stopwords (español, preservando: como, nada, ni, no, poco, sin, todo)
5. Lematización (spaCy `es_core_news_md`)

## Resultados Finales

### Clasificación de Sentimiento (Modelos Tradicionales)

| Modelo | Accuracy | F1-Macro | Recall |
|--------|----------|----------|--------|
| Logistic Regression | 95.48% | 0.944 | 0.948 |
| SVM | 95.98% | 0.948 | 0.951 |
| **Random Forest** | **95.98%** | **0.952** | **0.963** |

**Mejor modelo**: Random Forest (criterio: F1-Macro para clases desbalanceadas)

### Clasificación de Tipo de Queja (Redes Neuronales)

| Modelo | Accuracy | F1-Weighted | Recall |
|--------|----------|-------------|--------|
| **FNN** | **96.98%** | **0.970** | **0.935** |
| CNN | 93.47% | 0.928 | 0.809 |
| LSTM | 69.35% | 0.666 | 0.494 |

**Mejor modelo**: FNN (criterio: F1-Weighted para distribución muy desbalanceada)

### Rendimiento por Clase (Mejores Modelos)

#### Random Forest → Sentimiento
| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negativo | 0.98 | 0.99 | 0.98 |
| Neutro | 0.87 | 1.00 | 0.93 |
| Positivo | 0.98 | 0.90 | 0.94 |

#### FNN → Tipo de Queja
| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Por precio | 1.00 | 0.97 | 0.98 |
| Ninguna | 0.96 | 0.99 | 0.98 |
| Por alimentos | 0.96 | 0.96 | 0.96 |
| Por infraestructura | 1.00 | 0.82 | 0.90 |

## Conclusiones

### Hallazgos Principales

1. **Modelos tradicionales son suficientes para sentimiento**: TF-IDF + Random Forest logra 96% de accuracy con excelente balance entre clases. No se requieren redes neuronales complejas para esta tarea.

2. **FNN supera a arquitecturas más complejas para quejas**: Una red feedforward simple con embeddings de spaCy (300 dims) alcanza 97% de accuracy, superando a CNN (93%) y LSTM (69%).

3. **LSTM no es adecuado para vectores agregados**: La arquitectura LSTM está diseñada para secuencias de texto, no para vectores densos precomputados. Presenta underfitting severo en esta configuración.

4. **Métricas apropiadas por tarea**:
   - **F1-Macro** para sentimiento: La clase "neutro" (17% de muestras) requiere igual peso en la evaluación.
   - **F1-Weighted** para quejas: La clase "Ninguna" (52%) domina; se pondera por frecuencia real.

5. **Sin overfitting**: Los modelos ganadores muestran gaps train-test menores al 4%, indicando buena generalización.

### Arquitecturas Seleccionadas

| Tarea | Modelo | Entrada | Métricas |
|-------|--------|---------|----------|
| Sentimiento | Random Forest | TF-IDF (1,2-gramas) | Acc=96%, F1=0.95 |
| Quejas | FNN (2 capas) | spaCy vectors (300d) | Acc=97%, F1=0.97 |

### Limitaciones y Trabajo Futuro

- **Clase minoritaria "Por infraestructura"** (5.5%): Recall de 82% podría mejorarse con técnicas de oversampling o class weighting.
- **Dataset sintético**: Resultados deben validarse con datos reales de producción.
- **LSTM alternativo**: Podría funcionar mejor usando secuencias de palabras con capa Embedding en lugar de vectores agregados.

## Pasos Completados
1. ☑️ Estructura de carpetas
2. ☑️ Módulo de preprocesamiento (texto + vectorización spaCy)
3. ☑️ Implementar modelos tradicionales (Logistic, SVM, RF)
4. ☑️ Implementar redes neuronales para sentimiento (FNN, CNN, LSTM)
5. ☑️ Integrar clasificador de quejas (CNN con embeddings spaCy)
6. ☑️ Notebook de comparación para ambas tareas
7. ☑️ Ejecutar comparación completa y documentar conclusiones
