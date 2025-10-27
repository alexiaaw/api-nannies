"""
FilterService:
- filter_and_score(df, filters): realiza filtrado exacto en zone y availability,
  aplica coincidencias parciales en lists (qualities/courses/career), y si hay múltiples resultados,
  usa el modelo (predict_proba) para puntuar y ordenar.
"""

from pathlib import Path
import joblib
import config
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

MODEL_PATH = Path(config.NANNY_MODEL_PATH)

class FilterService:
    def __init__(self):
        self.model_payload = None
        if MODEL_PATH.exists():
            self._load_model()

    def _load_model(self):
        self.model_payload = joblib.load(MODEL_PATH)

    def _ensure_model_loaded(self):
        if self.model_payload is None:
            raise FileNotFoundError("Modelo no encontrado. Enviar datos a /api/nannies primero para entrenarlo.")

    def filter_and_score(self, df: pd.DataFrame, filters: dict) -> dict:
        """
        Realiza filtrado y devuelve dict con 'count' y 'nannies' (lista).
        Si hay múltiples coincidencias, calcula 'score' usando el modelo (probabilidad de availability).
        """
        # Filtrado exacto en zone y availability
        zone = filters.get("zone", "").strip().lower()
        availability = filters.get("availability", True)
        df_filtered = df[
            (df["zone"].str.strip().str.lower() == zone) &
            (df["availability"] == availability)
        ].copy()

        # Filtrado parcial en lists: si filters proporcionan elementos, se acepta si hay intersección
        for col in ("qualities", "courses", "career"):
            vals = filters.get(col, [])
            if vals:
                df_filtered = df_filtered[df_filtered[col].apply(lambda lst: bool(set(lst) & set(vals)))]

        # Si no hay resultados, retornar vacío
        if df_filtered.empty:
            return {"count": 0, "nannies": []}

        # Preparar resultados
        df_filtered = df_filtered.copy()

        # Si hay múltiples, usar modelo para puntuar
        if len(df_filtered) > 1:
            self._ensure_model_loaded()
            model = self.model_payload["model"]
            transformers = self.model_payload["transformers"]
            X = self._transform_df_to_X(df_filtered, transformers)
            if hasattr(model, "predict_proba"):
                df_filtered["score"] = model.predict_proba(X)[:, 1]
            else:
                df_filtered["score"] = model.predict(X).astype(float)
            df_filtered = df_filtered.sort_values("score", ascending=False)
        else:
            df_filtered["score"] = 1.0  # único resultado, score máximo

        # Convertir a lista de dicts
        results = []
        for _, row in df_filtered.iterrows():
            results.append({
                "id": row["id"],
                "name": row.get("name") or "Sin nombre",
                "qualities": row["qualities"],
                "courses": row["courses"],
                "career": row["career"],
                "zone": row["zone"],
                "availability": bool(row["availability"]),
                "score": float(row["score"])
            })

        return {"count": len(results), "nannies": results}


    def _transform_df_to_X(self, df, transformers):
        """
        Dado df y transformers guardados, construir X compatible con el modelo.
        """
        mlb_q = transformers["mlb_q"]
        mlb_c = transformers["mlb_c"]
        mlb_ca = transformers["mlb_ca"]
        zone_cols = transformers["zone_cols"]  # ejemplo: ['zone__guadalajara', ...]
        # Transform lists
        Q = mlb_q.transform(df["qualities"])
        C = mlb_c.transform(df["courses"])
        CA = mlb_ca.transform(df["career"])
        # Zones -> create array with columns zone_cols
        zone_names = [z.replace("zone__", "") for z in zone_cols]
        Z = np.zeros((len(df), len(zone_names)), dtype=int)
        zone_index = {z: i for i, z in enumerate(zone_names)}
        for i, z in enumerate(df["zone"].astype(str)):
            if z in zone_index:
                Z[i, zone_index[z]] = 1
            else:
                # zone not seen at training time -> all zeros (or handle differently)
                pass
        X = np.hstack([Q, C, CA, Z])
        return X
