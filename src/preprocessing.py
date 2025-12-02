"""
Módulo de preprocesamiento de texto para clasificación de sentimientos y quejas.
"""
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Descargar recursos de NLTK (solo la primera vez)
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Cargar recursos
download_nltk_resources()

# Configurar stopwords en español (preservando palabras con carga semántica)
STOPWORDS_ES = stopwords.words('spanish')
PRESERVE_WORDS = ['como', 'nada', 'ni', 'no', 'poco', 'sin', 'todo']
for word in PRESERVE_WORDS:
    if word in STOPWORDS_ES:
        STOPWORDS_ES.remove(word)

# Cargar modelo de spaCy para lematización
try:
    LEMMATIZER = spacy.load("es_core_news_md")
except OSError:
    print("Descargando modelo de spaCy...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_md"])
    LEMMATIZER = spacy.load("es_core_news_md")


def preprocess_text(text: str) -> str:
    """
    Preprocesa un texto para clasificación de sentimientos.
    
    Pasos:
    1. Convertir a minúsculas
    2. Eliminar caracteres especiales (mantener letras, números, acentos)
    3. Tokenizar
    4. Eliminar stopwords
    5. Lematizar
    
    Args:
        text: Texto a preprocesar
        
    Returns:
        Texto preprocesado
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Limpiar caracteres especiales
    text = re.sub(r'[^a-z0-9áéíóúüñ \t]', ' ', text)
    
    # Tokenizar
    tokens = word_tokenize(text)
    
    # Eliminar stopwords
    tokens = [t for t in tokens if t not in STOPWORDS_ES]
    
    # Lematizar
    doc = LEMMATIZER(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc if token.lemma_.strip()]
    
    return " ".join(lemmas)


def preprocess_dataframe(df, text_column: str = 'comment', output_column: str = 'clean_comment'):
    """
    Aplica preprocesamiento a una columna de un DataFrame.
    
    Args:
        df: DataFrame de pandas
        text_column: Nombre de la columna con el texto original
        output_column: Nombre de la columna para el texto preprocesado
        
    Returns:
        DataFrame con la nueva columna de texto preprocesado
    """
    from tqdm import tqdm
    tqdm.pandas()
    
    # Eliminar filas con texto vacío
    df = df.dropna(subset=[text_column])
    
    # Aplicar preprocesamiento
    df[output_column] = df[text_column].progress_apply(preprocess_text)
    
    return df


def vectorize_text(text: str, vector_size: int = 300) -> np.ndarray:
    """
    Convierte texto a vector usando embeddings de spaCy.
    
    Args:
        text: Texto a vectorizar
        vector_size: Dimensión del vector (default 300 para es_core_news_md)
        
    Returns:
        Vector numpy de dimensión vector_size
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(vector_size)
    
    # Preprocesar
    text = text.lower()
    text = re.sub(r'[^a-z0-9áéíóúüñ \t]', ' ', text)
    
    # Procesar con spaCy
    doc = LEMMATIZER(text)
    tokens = [t for t in doc if not t.is_stop and not t.is_punct]
    lemmas = [t.lemma_ for t in tokens]
    clean_text = " ".join(lemmas)
    
    # Obtener vector
    vector = LEMMATIZER(clean_text).vector
    return vector


def vectorize_dataframe(df, text_column: str = 'comment', output_column: str = 'vector'):
    """
    Aplica vectorización a una columna de un DataFrame.
    
    Args:
        df: DataFrame de pandas
        text_column: Nombre de la columna con el texto original
        output_column: Nombre de la columna para los vectores
        
    Returns:
        DataFrame con la nueva columna de vectores
    """
    from tqdm import tqdm
    tqdm.pandas()
    
    df = df.dropna(subset=[text_column])
    df[output_column] = df[text_column].progress_apply(vectorize_text)
    
    return df


def prepare_complaint_labels(df, complaint_column: str = 'complaint') -> tuple:
    """
    Prepara etiquetas para clasificación de tipo de queja.
    
    Mapeo:
        - precio -> 0
        - ninguna (NaN) -> 1  
        - sabor/variedad/alimentos -> 2
        - instalaciones/infraestructura -> 3
    
    Args:
        df: DataFrame con columna de quejas
        complaint_column: Nombre de la columna de quejas
        
    Returns:
        Tuple (df con columna 'complaint_category', mapeo de categorías)
    """
    # Mapeo de quejas a categorías
    complaint_mapping = {
        'precio': 0,
        'sabor': 2,
        'variedad': 2,
        'instalaciones': 3,
        'infraestructura': 3
    }
    
    category_names = {
        0: 'Por precio',
        1: 'Ninguna',
        2: 'Por alimentos',
        3: 'Por infraestructura'
    }
    
    def map_complaint(complaint):
        if pd.isna(complaint) or complaint == '':
            return 1  # Ninguna
        complaint = str(complaint).lower().strip()
        return complaint_mapping.get(complaint, 1)
    
    df = df.copy()
    df['complaint_category'] = df[complaint_column].apply(map_complaint)
    
    return df, category_names
