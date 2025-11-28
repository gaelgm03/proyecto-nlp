# Clasificación de Sentimientos en Encuestas de Satisfacción

## Descripción general del proyecto
Aplicación de Procesamiento de Lenguaje Natural (NLP) que analiza respuestas abiertas de encuestas de satisfacción y clasifica el sentimiento expresado. El sistema compara dos enfoques de Machine Learning para identificar qué tan bien se distingue entre opiniones positivas, negativas y neutras.

## Objetivo principal
Construir, evaluar y documentar dos modelos de clasificación de sentimientos que ayuden a entender el estado de ánimo de los participantes de las encuestas, facilitando la toma de decisiones académicas o administrativas.

## Tecnologías y herramientas utilizadas
- Python 3.11+
- Bibliotecas de NLP y ML: scikit-learn, pandas, numpy, nltk/spacy (por definir exactamente)
- Framework de redes neuronales: TensorFlow o PyTorch (a confirmar según experimentación)
- Jupyter Notebooks para experimentos
- Git para control de versiones

## Descripción del dataset
- Fuente: conjunto interno de encuestas de satisfacción estudiantil (respuestas de texto libre).
- Tamaño estimado: cientos a pocos miles de registros.
- Campos principales: identificador de encuesta, texto de la respuesta, etiqueta de sentimiento.
- Datos anonimizados y balanceados de manera preliminar para asegurar representatividad.

## Estructura general del proyecto
```
proyecto-nlp/
├── data/                # Dataset original y derivados (no versionados en Git)
├── notebooks/           # Exploración y experimentos iniciales
├── src/                 # Código fuente reutilizable
│   ├── traditional/     # Modelo clásico (ej. SVM, Logistic Regression)
│   └── neural/          # Modelo basado en redes neuronales
├── models/              # Pesos entrenados y artefactos exportados
└── README.md            # Documentación principal
```

## Cómo ejecutar el proyecto
1. Crear un entorno virtual: `python -m venv .venv` y activarlo.
2. Instalar dependencias (archivo `requirements.txt` pendiente): `pip install -r requirements.txt`.
3. Colocar el dataset en `data/` siguiendo las instrucciones internas (por definir).
4. Ejecutar experimentos desde los notebooks o correr `python src/traditional/train.py` y `python src/neural/train.py` cuando estén disponibles.

## Estado actual del proyecto
- Versión inicial (v0.1).
- Estructura del repositorio y documentación preliminar.
- Configuración del entorno en progreso.

## Trabajo futuro planeado
1. Definir y documentar el pipeline de preprocesamiento textual.
2. Implementar y entrenar el modelo tradicional con sus métricas clave.
3. Desarrollar la arquitectura neuronal (ej. LSTM o Transformer ligero).
4. Crear scripts reproducibles de entrenamiento y evaluación.
5. Incorporar visualizaciones y reporte comparativo entre modelos.
6. Publicar conjunto mínimo de pruebas unitarias/end-to-end.
