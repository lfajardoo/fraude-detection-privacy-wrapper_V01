"""
Script para entrenar un modelo simple de detección de fraude.
Modelo educativo - usa solo una muestra del dataset para rapidez.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    print("Cargando dataset...")
    # Cargar dataset
    df = pd.read_csv('dataset/creditcard.csv')

    print(f"Dataset original: {len(df)} registros")
    print(f"Fraudes: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")

    # Para hacer el entrenamiento rápido, usar solo una muestra
    # Balancear un poco las clases para mejor aprendizaje
    fraud = df[df['Class'] == 1].sample(min(500, df['Class'].sum()), random_state=42)
    normal = df[df['Class'] == 0].sample(5000, random_state=42)
    df_sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

    print(f"Muestra de entrenamiento: {len(df_sample)} registros")
    print(f"Fraudes en muestra: {df_sample['Class'].sum()}")

    # Separar características y target
    # Usar solo algunas características para simplicidad
    feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                    'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'Amount']

    X = df_sample[feature_cols]
    y = df_sample['Class']

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nEntrenando modelo (Regresión Logística)...")
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar modelo simple
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    # Evaluar
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"Accuracy en entrenamiento: {train_score:.4f}")
    print(f"Accuracy en test: {test_score:.4f}")

    # Guardar modelo y scaler
    print("\nGuardando modelo...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/features.pkl')

    print("✓ Modelo entrenado y guardado exitosamente!")
    print(f"  - models/fraud_model.pkl")
    print(f"  - models/scaler.pkl")
    print(f"  - models/features.pkl")

    return model, scaler, feature_cols

if __name__ == "__main__":
    train_model()

