import os
import pandas as pd
from pathlib import Path
import config

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame):
    """
    Guarda el dataframe (último dataset) en DATA_CSV_PATH.
    """
    path = Path(config.DATA_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_dataframe() -> pd.DataFrame:
    """
    Lee el último dataset guardado; si no existe, devuelve None.
    """
    path = Path(config.DATA_CSV_PATH)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"id": object})
    # columnas con listas: qualities, courses, career -> están guardadas como strings
    # convertir de string a listas si fuese necesario
    import ast
    def parse_list(val):
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed]
        except Exception:
            # intentar con split por comas
            return [s.strip().lower() for s in str(val).split(",") if s.strip()]
        return []
    for col in ["qualities","courses","career"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
    if "availability" in df.columns:
        df["availability"] = df["availability"].astype(bool)
    if "zone" in df.columns:
        df["zone"] = df["zone"].str.strip().str.lower()
    return df
