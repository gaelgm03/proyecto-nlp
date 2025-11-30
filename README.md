# Clasificación de Sentimientos en Encuestas de Satisfacción

## Descripción
Proyecto de NLP para clasificar sentimientos (positivo, neutro, negativo) en comentarios de encuestas de satisfacción. Compara modelos tradicionales de ML con redes neuronales.

## Dataset
- **Fuente**: Encuestas de satisfacción de cafetería universitaria (sintéticas generadas con GPT)
- **Tamaño**: ~1000 registros
- **Columnas**: `role`, `genre`, `age`, `comment`, `kind_of_comment`, `complaint`
- **Clases**: positivo, neutro, negativo

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
│   └── 01_model_comparison.ipynb  # Comparación de modelos
├── src/                           # Código fuente
│   ├── preprocessing.py           # Limpieza y preprocesamiento de texto
│   ├── traditional/               # Modelos tradicionales
│   │   └── models.py
│   └── neural/                    # Redes neuronales
│       └── models.py
├── models/                        # Modelos entrenados
├── Create_DB.ipynb               # Generación de datos sintéticos
├── Kind_Classifier.ipynb         # Experimentos iniciales (SVM)
├── requirements.txt              # Dependencias
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
from src.preprocessing import preprocess_dataframe
from src.traditional.models import train_and_compare_models
from src.neural.models import train_and_compare_neural_models
import pandas as pd

# Cargar y preprocesar datos
df = pd.read_csv('data/respuestas_cafeteria.csv')
df = preprocess_dataframe(df)

X = df['clean_comment']
y = df['kind_of_comment']

# Modelos tradicionales
trad_results, _ = train_and_compare_models(X, y)

# Redes neuronales
neural_results, _ = train_and_compare_neural_models(X, y, epochs=30)
```

### 2. Usar un modelo específico
```python
from src.traditional.models import TraditionalClassifier

# SVM
clf = TraditionalClassifier(model_type='svm')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
clf.save('models/svm_model.joblib')
```

## Pipeline de Preprocesamiento
1. Convertir a minúsculas
2. Eliminar caracteres especiales (mantener acentos)
3. Tokenización (NLTK)
4. Eliminación de stopwords (español, preservando: como, nada, ni, no, poco, sin, todo)
5. Lematización (spaCy `es_core_news_md`)

## Resultados Baseline (SVM)
- **Accuracy**: 96.5%
- **F1-score negativo**: 0.99
- **F1-score neutro**: 0.91
- **F1-score positivo**: 0.96

## Próximos Pasos
1. ☑️ Estructura de carpetas
2. ☑️ Módulo de preprocesamiento
3. ☑️ Implementar modelos tradicionales (Logistic, SVM, RF)
4. ☑️ Implementar redes neuronales (FNN, CNN, LSTM)
5. ⬜ Ejecutar comparación completa
6. ⬜ Documentar conclusiones finales
