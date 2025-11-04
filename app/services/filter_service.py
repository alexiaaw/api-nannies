"""
FilterService mejorado:
- Devuelve siempre 3 niñeras (mezclando exactas y similares).
- Zona y disponibilidad son obligatorias para coincidir.
- Calcula coincidencia parcial en cualidades, cursos y carrera.
- Si hay menos de 3 exactas, completa con las más similares.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import config


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
            raise FileNotFoundError("Modelo no encontrado. Entrena primero con /api/nannies.")

    def filter_and_score(self, df: pd.DataFrame, filters: dict) -> dict:
        zone = filters.get("zone", "").strip().lower()
        availability = filters.get("availability", True)
        qualities = set(filters.get("qualities", []))
        courses = set(filters.get("courses", []))
        career = set(filters.get("career", []))

        # 🔹 1. Filtrar solo por zona y disponibilidad (obligatorio)
        df_zone = df[
            (df["zone"].str.strip().str.lower() == zone) &
            (df["availability"] == availability)
        ].copy()

        if df_zone.empty:
            return {
                "mode": "no_zone_match",
                "count": 0,
                "nannies": []
            }

        # 🔹 2. Calcular coincidencia parcial por atributos
        df_zone["match_score"] = df_zone.apply(
            lambda row: self._calculate_match_score(
                row, qualities, courses, career
            ),
            axis=1
        )

        # 🔹 3. Ordenar por score y tomar las 3 mejores coincidencias exactas
        df_exact_top = df_zone.sort_values("match_score", ascending=False).head(3)

        # 🔹 4. Si faltan niñeras para completar 3 → buscar similares (dentro de la misma zona)
        if len(df_exact_top) < 3:
            needed = 3 - len(df_exact_top)
            remaining = df_zone[~df_zone["id"].isin(df_exact_top["id"])].copy()
            if not remaining.empty:
                remaining["similarity_score"] = remaining.apply(
                    lambda row: self._calculate_similarity(
                        row, qualities, courses, career
                    ),
                    axis=1
                )
                df_similar = remaining.sort_values("similarity_score", ascending=False).head(needed)
                df_final = pd.concat([df_exact_top, df_similar]).head(3)
            else:
                # Si no hay más en la zona, tomar de otras zonas similares
                others = df[df["availability"] == availability].copy()
                others["similarity_score"] = others.apply(
                    lambda row: self._calculate_similarity(
                        row, qualities, courses, career
                    ),
                    axis=1
                )
                df_others = others.sort_values("similarity_score", ascending=False).head(needed)
                df_final = pd.concat([df_exact_top, df_others]).head(3)
        else:
            df_final = df_exact_top.copy()

        # 🔹 5. Scoring con modelo (si existe)
        try:
            self._ensure_model_loaded()
            model = self.model_payload["model"]
            transformers = self.model_payload["transformers"]
            X = self._transform_df_to_X(df_final, transformers)
            if hasattr(model, "predict_proba"):
                df_final["model_score"] = model.predict_proba(X)[:, 1]
            else:
                df_final["model_score"] = model.predict(X).astype(float)
        except Exception:
            df_final["model_score"] = df_final["match_score"] / 100  # fallback

        # 🔹 6. Construir respuesta
        results = []
        for _, row in df_final.iterrows():
            results.append({
                "id": row["id"],
                "name": row.get("name", "Sin nombre"),
                "zone": row["zone"],
                "availability": bool(row["availability"]),
                "qualities": row["qualities"],
                "courses": row["courses"],
                "career": row["career"],
                "match_score": float(round(row["match_score"], 2)),
                "model_score": float(round(row["model_score"], 4))
            })

        return {
            "mode": "hybrid_filter",
            "count": len(results),
            "nannies": results
        }

    # -------------------------------
    # 🔧 Funciones auxiliares
    # -------------------------------
    def _calculate_match_score(self, row, qualities, courses, career):
        """Calcula coincidencia porcentual considerando los filtros definidos."""
        total = 0
        matches = 0

        if qualities:
            total += 1
            q_match = len(qualities.intersection(set(row["qualities"])))
            matches += q_match / len(qualities)

        if courses:
            total += 1
            c_match = len(courses.intersection(set(row["courses"])))
            matches += c_match / len(courses)

        if career:
            total += 1
            ca_match = len(career.intersection(set(row["career"])))
            matches += ca_match / len(career)

        return (matches / total * 100) if total > 0 else 0.0

    def _calculate_similarity(self, row, qualities, courses, career):
        """Calcula una puntuación simple de similitud para completar resultados."""
        score = 0
        if qualities:
            score += len(qualities.intersection(set(row["qualities"])))
        if courses:
            score += len(courses.intersection(set(row["courses"])))
        if career:
            score += len(career.intersection(set(row["career"])))
        return score

    def _transform_df_to_X(self, df, transformers):
        """Convierte el DataFrame en formato numérico para el modelo."""
        mlb_q = transformers["mlb_q"]
        mlb_c = transformers["mlb_c"]
        mlb_ca = transformers["mlb_ca"]
        zone_cols = transformers["zone_cols"]

        Q = mlb_q.transform(df["qualities"])
        C = mlb_c.transform(df["courses"])
        CA = mlb_ca.transform(df["career"])

        zone_names = [z.replace("zone__", "") for z in zone_cols]
        Z = np.zeros((len(df), len(zone_names)), dtype=int)
        zone_index = {z: i for i, z in enumerate(zone_names)}

        for i, z in enumerate(df["zone"].astype(str).str.lower()):
            if z in zone_index:
                Z[i, zone_index[z]] = 1

        return np.hstack([Q, C, CA, Z])
