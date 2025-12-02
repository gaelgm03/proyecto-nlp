"""
Modelos de redes neuronales para clasificación de tipo de queja.
Usa embeddings de spaCy (300 dims) como entrada.
Incluye: FNN, CNN, LSTM
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, 
    Flatten, LSTM, Input, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import os


# Nombres de categorías de queja
COMPLAINT_CATEGORIES = {
    0: 'Por precio',
    1: 'Ninguna',
    2: 'Por alimentos',
    3: 'Por infraestructura'
}


class ComplaintClassifier:
    """Clasificador de tipo de queja usando embeddings de spaCy."""
    
    def __init__(
        self,
        model_type: str = 'cnn',
        vector_size: int = 300,
        num_classes: int = 4,
        random_state: int = 42
    ):
        """
        Inicializa el clasificador de quejas.
        
        Args:
            model_type: 'fnn', 'cnn', o 'lstm'
            vector_size: Dimensión del vector de entrada (300 para spaCy)
            num_classes: Número de clases de queja
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.vector_size = vector_size
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _build_fnn(self) -> Sequential:
        """Red Feedforward simple."""
        model = Sequential([
            Input(shape=(self.vector_size,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_cnn(self) -> Sequential:
        """Red Convolucional 1D sobre vectores."""
        model = Sequential([
            Input(shape=(self.vector_size, 1)),
            Conv1D(30, 2, activation='relu'),
            MaxPooling1D(5),
            Flatten(),
            Dense(35, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_lstm(self) -> Sequential:
        """Red LSTM sobre vectores (optimizada para evitar underfitting)."""
        model = Sequential([
            Input(shape=(self.vector_size, 1)),
            LSTM(32, dropout=0.1, recurrent_dropout=0.1),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_model(self) -> Sequential:
        """Construye el modelo según el tipo especificado."""
        builders = {
            'fnn': self._build_fnn,
            'cnn': self._build_cnn,
            'lstm': self._build_lstm
        }
        
        if self.model_type not in builders:
            raise ValueError(f"Modelo no soportado: {self.model_type}")
        
        model = builders[self.model_type]()
        
        # Learning rate menor para LSTM (evita explosión de gradientes)
        lr = 0.001 if self.model_type == 'lstm' else 0.01
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None):
        """Prepara los datos para el modelo."""
        # Reshape para CNN/LSTM
        if self.model_type in ['cnn', 'lstm']:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        if y is not None:
            y_cat = to_categorical(y, num_classes=self.num_classes)
            return X, y_cat
        return X
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Entrena el modelo.
        
        Args:
            X_train: Vectores de entrenamiento (N, 300)
            y_train: Etiquetas de entrenamiento
            X_val: Vectores de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            epochs: Número máximo de épocas
            batch_size: Tamaño de batch
            verbose: Nivel de verbosidad
        """
        # Preparar datos
        X_train_prep, y_train_cat = self._prepare_data(X_train, y_train)
        
        # Construir modelo
        self.model = self._build_model()
        
        # Callbacks (patience mayor para LSTM que necesita más épocas)
        patience = 10 if self.model_type == 'lstm' else 5
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            )
        ]
        
        # Validación
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_prep, y_val_cat = self._prepare_data(X_val, y_val)
            validation_data = (X_val_prep, y_val_cat)
        
        # Entrenar
        history = self.model.fit(
            X_train_prep, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones (retorna índices de clase)."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")
        
        X_prep = self._prepare_data(X)
        y_pred_proba = self.model.predict(X_prep, verbose=0)
        return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades de predicción."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")
        
        X_prep = self._prepare_data(X)
        return self.model.predict(X_prep, verbose=0)
    
    def predict_labels(self, X: np.ndarray) -> list:
        """Retorna nombres de categorías predichas."""
        predictions = self.predict(X)
        return [COMPLAINT_CATEGORIES[p] for p in predictions]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evalúa el modelo y retorna métricas completas."""
        y_pred = self.predict(X_test)
        
        target_names = [COMPLAINT_CATEGORIES[i] for i in range(self.num_classes)]
        
        return {
            'model_type': self.model_type,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(
                y_test, y_pred, target_names=target_names
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred
        }
    
    def save(self, path: str):
        """Guarda el modelo."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        model_path = path.replace('.h5', '') + '_complaint_model.h5'
        self.model.save(model_path)
        print(f"Modelo guardado en: {model_path}")
    
    def load(self, path: str):
        """Carga el modelo."""
        model_path = path.replace('.h5', '') + '_complaint_model.h5'
        self.model = tf.keras.models.load_model(model_path)
        self.is_fitted = True
        print(f"Modelo cargado desde: {model_path}")
        return self


def train_and_compare_complaint_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32
):
    """
    Entrena y compara diferentes arquitecturas para clasificación de quejas.
    
    Args:
        X: Array de vectores (N, 300)
        y: Array de etiquetas
        test_size: Proporción para test
        val_size: Proporción para validación
        epochs: Número máximo de épocas
        batch_size: Tamaño de batch
        
    Returns:
        dict con resultados de cada modelo
    """
    # Split de datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    model_types = ['fnn', 'cnn', 'lstm']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Entrenando: {model_type.upper()} (Quejas)")
        print('='*50)
        
        clf = ComplaintClassifier(model_type=model_type)
        history = clf.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        eval_results = clf.evaluate(X_test, y_test)
        
        print(f"\nAccuracy: {eval_results['accuracy']:.4f}")
        print(eval_results['classification_report'])
        
        results[model_type] = {
            'classifier': clf,
            'results': eval_results,
            'history': history.history
        }
    
    return results, (X_train, X_val, X_test, y_train, y_val, y_test)
