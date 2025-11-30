"""
Modelos tradicionales de ML para clasificación de sentimientos.
Incluye: Logistic Regression, SVM (LinearSVC), Random Forest
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class TraditionalClassifier:
    """Clase base para clasificadores tradicionales de ML."""
    
    def __init__(self, model_type: str = 'svm', random_state: int = 42):
        """
        Inicializa el clasificador.
        
        Args:
            model_type: 'logistic', 'svm', o 'random_forest'
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9
        )
        self.is_fitted = False
        
    def _create_model(self):
        """Crea el modelo según el tipo especificado."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'svm':
            return LinearSVC(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=2000
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Modelo no soportado: {self.model_type}")
    
    def fit(self, X_train: pd.Series, y_train: pd.Series):
        """Entrena el modelo."""
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Realiza predicciones."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)
    
    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> dict:
        """Evalúa el modelo y retorna métricas."""
        y_pred = self.predict(X_test)
        
        return {
            'model_type': self.model_type,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred
        }
    
    def cross_validate(self, X: pd.Series, y: pd.Series, cv: int = 5) -> dict:
        """Realiza validación cruzada."""
        X_tfidf = self.vectorizer.fit_transform(X)
        scores = cross_val_score(self.model, X_tfidf, y, cv=cv, scoring='accuracy')
        
        return {
            'model_type': self.model_type,
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
    
    def save(self, path: str):
        """Guarda el modelo y vectorizador."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path: str):
        """Carga el modelo y vectorizador."""
        data = joblib.load(path)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.model_type = data['model_type']
        self.is_fitted = True
        print(f"Modelo cargado desde: {path}")
        return self


def train_and_compare_models(X: pd.Series, y: pd.Series, test_size: float = 0.2):
    """
    Entrena y compara los tres modelos tradicionales.
    
    Args:
        X: Serie con textos preprocesados
        y: Serie con etiquetas
        test_size: Proporción para test
        
    Returns:
        dict con resultados de cada modelo
    """
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    results = {}
    model_types = ['logistic', 'svm', 'random_forest']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Entrenando: {model_type.upper()}")
        print('='*50)
        
        clf = TraditionalClassifier(model_type=model_type)
        clf.fit(X_train, y_train)
        
        eval_results = clf.evaluate(X_test, y_test)
        
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(eval_results['classification_report'])
        
        results[model_type] = {
            'classifier': clf,
            'results': eval_results
        }
    
    return results, (X_train, X_test, y_train, y_test)
