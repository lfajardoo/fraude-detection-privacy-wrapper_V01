"""
Clase del modelo de fraude - carga y realiza predicciones
"""
import joblib
import numpy as np
import os

class FraudModel:
    def __init__(self, model_path='models'):
        """Inicializa y carga el modelo entrenado"""
        self.model = joblib.load(os.path.join(model_path, 'fraud_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        self.features = joblib.load(os.path.join(model_path, 'features.pkl'))

    def predict(self, transaction_data: dict) -> dict:
        """
        Predice si una transacción es fraudulenta

        Args:
            transaction_data: Diccionario con las características de la transacción

        Returns:
            Diccionario con la predicción y score de confianza
        """
        # Extraer características en el orden correcto
        features_values = []
        for feature in self.features:
            if feature not in transaction_data:
                raise ValueError(f"Característica faltante: {feature}")
            features_values.append(transaction_data[feature])

        # Convertir a array numpy
        X = np.array([features_values])

        # Escalar
        X_scaled = self.scaler.transform(X)

        # Predecir
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]

        # Probabilidad de fraude (clase 1)
        fraud_probability = probability[1]

        return {
            'is_fraud': int(prediction),
            'fraud_probability': float(fraud_probability),
            'confidence_score': float(max(probability))
        }

    def get_feature_names(self):
        """Retorna la lista de características requeridas"""
        return self.features

