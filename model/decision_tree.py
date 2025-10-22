import joblib, os
import pandas as pd
from model.pipeline import build_pipeline

class NannyDecisionTree:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline if pipeline is not None else build_pipeline()
        self.is_trained = False
        self.feature_names = ["zone", "career", "courses", "qualities"] 

# entrenamiento 
    def fit(self, X: pd.DataFrame, y):
        self.pipeline.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame):
        # Validación de estado
        if not self.is_trained:
            raise RuntimeError("El modelo debe ser entrenado o cargado antes de la predicción.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        # Validación de estado
        if not self.is_trained:
            raise RuntimeError("El modelo debe ser entrenado o cargado antes de la predicción.")
        return self.pipeline.predict_proba(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
        self.is_trained = True
        return self
