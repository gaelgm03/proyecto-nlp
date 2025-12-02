"""
Módulo de predicción usando los modelos ganadores (solo inferencia).

Modelos:
- Sentimiento: Random Forest (TF-IDF)
- Tipo de Queja: FNN (spaCy embeddings)

Nota: El entrenamiento se realiza en el notebook 01_model_comparison.ipynb
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union

from .preprocessing import preprocess_text, vectorize_text
from .traditional.models import TraditionalClassifier
from .neural.complaint_models import ComplaintClassifier, COMPLAINT_CATEGORIES


class SentimentPredictor:
    """Predictor de sentimiento usando modelo pre-entrenado."""
    
    def __init__(self, model_path: str):
        """
        Inicializa el predictor cargando un modelo guardado.
        
        Args:
            model_path: Ruta al modelo guardado (.joblib)
        """
        self.classifier = TraditionalClassifier(model_type='random_forest')
        self.is_loaded = False
        self.load(model_path)
    
    def predict(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predice el sentimiento de uno o más textos.
        
        Args:
            text: Texto único o lista de textos
            
        Returns:
            Predicción(es) de sentimiento
        """
        if not self.is_loaded:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        # Manejar texto único o lista
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        # Preprocesar
        processed = [preprocess_text(t) for t in texts]
        processed_series = pd.Series(processed)
        
        # Predecir
        predictions = self.classifier.predict(processed_series)
        
        return predictions[0] if single_input else list(predictions)
    
    def predict_with_confidence(self, text: str) -> Dict[str, float]:
        """
        Predice sentimiento con confianza por clase.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con probabilidades por clase
        """
        if not self.is_loaded:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        processed = preprocess_text(text)
        X_tfidf = self.classifier.vectorizer.transform([processed])
        
        # Random Forest tiene predict_proba
        if hasattr(self.classifier.model, 'predict_proba'):
            proba = self.classifier.model.predict_proba(X_tfidf)[0]
            classes = self.classifier.model.classes_
            return {cls: float(prob) for cls, prob in zip(classes, proba)}
        else:
            # Fallback para modelos sin probabilidades
            pred = self.classifier.predict(pd.Series([processed]))[0]
            return {pred: 1.0}
    
    def save(self, path: str):
        """Guarda el modelo."""
        self.classifier.save(path)
    
    def load(self, path: str):
        """Carga el modelo."""
        self.classifier.load(path)
        self.is_loaded = True


class ComplaintPredictor:
    """Predictor de tipo de queja usando modelo pre-entrenado."""
    
    def __init__(self, model_path: str):
        """
        Inicializa el predictor cargando un modelo guardado.
        
        Args:
            model_path: Ruta al modelo guardado (.h5)
        """
        self.classifier = ComplaintClassifier(model_type='fnn')
        self.is_loaded = False
        self.load(model_path)
    
    def predict(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predice el tipo de queja de uno o más textos.
        
        Args:
            text: Texto único o lista de textos
            
        Returns:
            Predicción(es) de tipo de queja
        """
        if not self.is_loaded:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        # Manejar texto único o lista
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        # Vectorizar
        vectors = np.array([vectorize_text(t) for t in texts])
        
        # Predecir
        predictions = self.classifier.predict_labels(vectors)
        
        return predictions[0] if single_input else predictions
    
    def predict_with_confidence(self, text: str) -> Dict[str, float]:
        """
        Predice tipo de queja con confianza por clase.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con probabilidades por categoría
        """
        if not self.is_loaded:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        vector = vectorize_text(text).reshape(1, -1)
        proba = self.classifier.predict_proba(vector)[0]
        
        return {COMPLAINT_CATEGORIES[i]: float(p) for i, p in enumerate(proba)}
    
    def save(self, path: str):
        """Guarda el modelo."""
        self.classifier.save(path)
    
    def load(self, path: str):
        """Carga el modelo."""
        self.classifier.load(path)
        self.is_loaded = True


class CafeteriaAnalyzer:
    """
    Analizador completo de comentarios de cafetería.
    Combina predicción de sentimiento y tipo de queja.
    Solo carga modelos pre-entrenados (no entrena).
    """
    
    def __init__(self, sentiment_model_path: str, complaint_model_path: str):
        """
        Inicializa el analizador cargando modelos pre-entrenados.
        
        Args:
            sentiment_model_path: Ruta al modelo de sentimiento (.joblib)
            complaint_model_path: Ruta al modelo de quejas (.h5)
        """
        self.sentiment_predictor = SentimentPredictor(sentiment_model_path)
        self.complaint_predictor = ComplaintPredictor(complaint_model_path)
    
    def analyze(self, text: str) -> Dict:
        """
        Analiza un comentario completo.
        
        Args:
            text: Comentario a analizar
            
        Returns:
            Diccionario con análisis completo
        """
        sentiment = self.sentiment_predictor.predict(text)
        sentiment_proba = self.sentiment_predictor.predict_with_confidence(text)
        
        complaint = self.complaint_predictor.predict(text)
        complaint_proba = self.complaint_predictor.predict_with_confidence(text)
        
        return {
            'texto_original': text,
            'sentimiento': {
                'prediccion': sentiment,
                'confianza': max(sentiment_proba.values()),
                'probabilidades': sentiment_proba
            },
            'tipo_queja': {
                'prediccion': complaint,
                'confianza': max(complaint_proba.values()),
                'probabilidades': complaint_proba
            }
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analiza múltiples comentarios.
        
        Args:
            texts: Lista de comentarios
            
        Returns:
            Lista de análisis
        """
        return [self.analyze(text) for text in texts]
    
    def save_models(self, sentiment_path: str, complaint_path: str):
        """Guarda ambos modelos."""
        self.sentiment_predictor.save(sentiment_path)
        self.complaint_predictor.save(complaint_path)
        print(f"\nModelos guardados:")
        print(f"  - Sentimiento: {sentiment_path}")
        print(f"  - Quejas: {complaint_path}")
    
    def load_models(self, sentiment_path: str, complaint_path: str):
        """Carga ambos modelos."""
        self.sentiment_predictor.load(sentiment_path)
        self.complaint_predictor.load(complaint_path)


def demo_predictions():
    """Demostración de predicciones con textos de ejemplo."""
    
    # Textos de ejemplo
    examples = [
        "La comida está muy rica y el servicio es excelente",
        "El precio es demasiado caro para lo que ofrecen",
        "Las instalaciones están sucias y descuidadas", 
        "Todo bien, normal, sin quejas",
        "La variedad de platillos es muy limitada",
        "Me encanta el ambiente, siempre vengo aquí",
        "El sabor de la comida ha empeorado mucho últimamente"
    ]
    
    print("\n" + "="*70)
    print("DEMOSTRACIÓN DE PREDICCIONES")
    print("="*70)
    
    for i, text in enumerate(examples, 1):
        print(f"\n[{i}] \"{text}\"")
        print("-" * 60)
    
    return examples


# Ejemplo de uso rápido
if __name__ == "__main__":
    print("Módulo de predicción (solo inferencia).")
    print("\nUso:")
    print("  from src.predict import CafeteriaAnalyzer")
    print("  analyzer = CafeteriaAnalyzer(")
    print("      sentiment_model_path='models/sentiment_best_random_forest.joblib',")
    print("      complaint_model_path='models/complaint_best_fnn.h5'")
    print("  )")
    print("  result = analyzer.analyze('La comida está muy rica')")
