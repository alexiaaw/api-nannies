"""
TrainerService:
- train_or_load(df): si hay modelo en NANNY_MODEL_PATH lo carga (y carga transformers),
  si no, entrena un DecisionTreeClassifier con el dataset recibido y lo guarda.
- Usamos availability como target (proxy). Guardamos también los transformers.
"""

import os
import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import config
from . import trainer_utils
from ..models.transformers import fit_transformers
from pathlib import Path

MODEL_DIR = Path(config.NANNY_MODEL_PATH).parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class TrainerService:
    def __init__(self):
        self.model_path = config.NANNY_MODEL_PATH

    def train_or_load(self, df: pd.DataFrame):
        """
        Si existe modelo, lo carga. Siempre reentrena con el nuevo dataframe recibido
        (especificación: 'Entrenamiento automático cada vez que se reciba un dataset nuevo').
        """
        # Preprocess and train
        X, y, transformers = trainer_utils.prepare_features_and_target(df)
        # DecisionTreeClassifier obligatorio
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        # Guardar modelo + transformers usando joblib
        payload = {
            "model": clf,
            "transformers": transformers
        }
        joblib.dump(payload, self.model_path)
        return True

    def load(self):
        """
        Carga el modelo y transformers desde disco; devuelve dict con model+transformers.
        """
        if not Path(self.model_path).exists():
            return None
        payload = joblib.load(self.model_path)
        return payload
