import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # carpeta src/
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')


def cargar_modelo():
    """Carga el modelo, el scaler y la metadata desde ../models/"""
    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    return model, scaler, metadata


def predecir(datos, model, scaler, metadata):
    """
    datos: dict con las 8 features del paciente, ej:
           {'Pregnancies': 2, 'Glucose': 130, ...}
    Devuelve un dict con la predicción, probabilidad y umbral usado.
    """
    feature_names = metadata['feature_names']
    threshold = metadata['threshold']

    X = np.array([[datos[feat] for feat in feature_names]])
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1][0]
    prediccion = int(proba >= threshold)

    return {
        'prediccion': prediccion,
        'probabilidad': round(float(proba), 4),
        'umbral_usado': threshold
    }