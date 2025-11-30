"""
Módulo de preprocesamiento de texto para clasificación de sentimientos.
"""
import re
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
