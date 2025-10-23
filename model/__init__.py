# api/model/__init__.py
import os
from .decision_tree import NannyDecisionTree

MODEL_PATH = os.environ.get("NANNY_MODEL_PATH", "saved_models/nanny_tree.pkl")

model = NannyDecisionTree()

# Intentar cargar el modelo guardado (si existe)
if os.path.exists(MODEL_PATH):
    try:
        model.load(MODEL_PATH)
        print(f"Modelo cargado correctamente desde: {MODEL_PATH}")
    except Exception as e:
        print(f"No se pudo cargar el modelo: {e}")
else:
    print("No se encontró modelo entrenado, se debe entrenar antes de predecir.")
