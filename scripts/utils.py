# scripts/utils.py
import pandas as pd

def load_nannies(path="data/nannies.csv"):
    df = pd.read_csv(path)
    # Limpiar columnas por si hay espacios
    for col in ["carreras","cursos","cualidades","zona","horario"]:
        df[col] = df[col].astype(str).str.strip()
    return df
