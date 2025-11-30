"""
Modelos de redes neuronales para clasificación de sentimientos.
Incluye: FNN (Feedforward), CNN, RNN (LSTM)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Embedding, LSTM,
    Conv1D, GlobalMaxPooling1D,
    Input, Flatten
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os


class NeuralClassifier:
    """Clase base para clasificadores de redes neuronales."""
    
    def __init__(
        self, 
        model_type: str = 'lstm',
        max_words: int = 10000,
        max_len: int = 100,
        embedding_dim: int = 128,
        random_state: int = 42
    ):
        """
        Inicializa el clasificador neuronal.
        
        Args:
            model_type: 'fnn', 'cnn', o 'lstm'
            max_words: Vocabulario máximo
            max_len: Longitud máxima de secuencia
            embedding_dim: Dimensión de embeddings
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        
        self.tokenizer = Tokenizer(num_words=max_words)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.num_classes = None
        self.is_fitted = False
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _build_fnn(self) -> Model:
        """Red Feedforward (FNN)."""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_cnn(self) -> Model:
        """Red Convolucional (CNN)."""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_lstm(self) -> Model:
        """Red LSTM."""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_model(self) -> Model:
        """Construye el modelo según el tipo especificado."""
        builders = {
            'fnn': self._build_fnn,
            'cnn': self._build_cnn,
            'lstm': self._build_lstm
        }
        
        if self.model_type not in builders:
            raise ValueError(f"Modelo no soportado: {self.model_type}")
        
        model = builders[self.model_type]()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _prepare_data(self, X: pd.Series, y: pd.Series = None, fit: bool = False):
        """Prepara los datos para entrenamiento/predicción."""
        if fit:
            self.tokenizer.fit_on_texts(X)
        
        sequences = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(sequences, maxlen=self.max_len)
        
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
                self.num_classes = len(self.label_encoder.classes_)
            else:
                y_encoded = self.label_encoder.transform(y)
            y_cat = to_categorical(y_encoded, num_classes=self.num_classes)
            return X_padded, y_cat
        
        return X_padded
    
    def fit(
        self, 
        X_train: pd.Series, 
        y_train: pd.Series,
        X_val: pd.Series = None,
        y_val: pd.Series = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """Entrena el modelo."""
        # Preparar datos de entrenamiento
        X_train_pad, y_train_cat = self._prepare_data(X_train, y_train, fit=True)
        
        # Construir modelo
        self.model = self._build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Preparar datos de validación
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_pad, y_val_cat = self._prepare_data(X_val, y_val, fit=False)
            validation_data = (X_val_pad, y_val_cat)
        
        # Entrenar
        history = self.model.fit(
            X_train_pad, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Realiza predicciones."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")
        
        X_padded = self._prepare_data(X, fit=False)
        y_pred_proba = self.model.predict(X_padded, verbose=0)
        y_pred_idx = np.argmax(y_pred_proba, axis=1)
        return self.label_encoder.inverse_transform(y_pred_idx)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Retorna probabilidades de predicción."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")
        
        X_padded = self._prepare_data(X, fit=False)
        return self.model.predict(X_padded, verbose=0)
    
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
    
    def save(self, path: str):
        """Guarda el modelo completo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar modelo Keras
        model_path = path.replace('.h5', '') + '_model.h5'
        self.model.save(model_path)
        
        # Guardar tokenizer y label encoder
        import pickle
        meta_path = path.replace('.h5', '') + '_meta.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'max_len': self.max_len,
                'max_words': self.max_words,
                'embedding_dim': self.embedding_dim,
                'num_classes': self.num_classes
            }, f)
        
        print(f"Modelo guardado en: {model_path}")
    
    def load(self, path: str):
        """Carga el modelo completo."""
        model_path = path.replace('.h5', '') + '_model.h5'
        meta_path = path.replace('.h5', '') + '_meta.pkl'
        
        # Cargar modelo Keras
        self.model = tf.keras.models.load_model(model_path)
        
        # Cargar metadatos
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.tokenizer = meta['tokenizer']
        self.label_encoder = meta['label_encoder']
        self.model_type = meta['model_type']
        self.max_len = meta['max_len']
        self.max_words = meta['max_words']
        self.embedding_dim = meta['embedding_dim']
        self.num_classes = meta['num_classes']
        self.is_fitted = True
        
        print(f"Modelo cargado desde: {model_path}")
        return self


def train_and_compare_neural_models(
    X: pd.Series, 
    y: pd.Series, 
    test_size: float = 0.2,
    val_size: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32
):
    """
    Entrena y compara diferentes arquitecturas de redes neuronales.
    
    Args:
        X: Serie con textos preprocesados
        y: Serie con etiquetas
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
        print(f"Entrenando: {model_type.upper()}")
        print('='*50)
        
        clf = NeuralClassifier(model_type=model_type)
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
