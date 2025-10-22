import pandas as pd
from model.decision_tree import NannyDecisionTree
from config import MODEL_PATH

def main():
    model = NannyDecisionTree()
    model.load(MODEL_PATH)

    example = {
        "zone": "Zona Centro",
        "career": ["Pedagogía"],
        "courses": ["Primeros Auxilios"],
        "qualities": ["Paciente"]
    }

    feature_cols = ["zone", "career", "courses", "qualities"]
    df = pd.DataFrame([example])[feature_cols]
    

    pred = model.predict(df)[0]
    try:
        proba = model.predict_proba(df)[0][1]
    except Exception:
        # Nota: Si el modelo no puede predecir, 1.0 es un valor arbitrario. 
        # Podrías querer usar 0.5 o None para ser más neutral.
        proba = 1.0 
        
    print("Predicción:", int(pred), "Prob:", round(float(proba), 3))

if __name__ == "__main__":
    main()