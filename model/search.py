# model/search.py
import pandas as pd
from utils.preprocessing import normalize_nannies_list

def normalize_text(s):
    return str(s).strip().lower() if s is not None else ""

def search_nannies(nannies_list, filters, model=None, top_k=None):
   
    nannies = normalize_nannies_list(nannies_list)

    # Filtros obligatorios
    user_zone = normalize_text(filters.get("zone", ""))
    if not user_zone:
        raise ValueError("Filtro 'zone' es obligatorio.")

    require_available = filters.get("availability", True) is True

    # Filtros opcionales
    optional_filters = {
        "career": filters.get("career", []),
        "courses": filters.get("courses", []),
        "qualities": filters.get("qualities", [])
    }

    # Normalizar filtros opcionales
    for key in optional_filters:
        values = optional_filters[key]
        if isinstance(values, str):
            values = [values]
        optional_filters[key] = [normalize_text(v) for v in values]

    results = []

    for nanny in nannies:
        # Validación obligatoria 
        if require_available and not nanny.get("availability", False):
            continue

        nanny_zones = nanny.get("zone")
        if isinstance(nanny_zones, str):
            nanny_zones = [nanny_zones]
        nanny_zones = [normalize_text(z) for z in (nanny_zones or [])]

        if user_zone not in nanny_zones:
            continue

        # Validación opcional 
        match_count = 0
        for key, values in optional_filters.items():
            if not values:
                continue
            nanny_values = nanny.get(key)
            if isinstance(nanny_values, str):
                nanny_values = [nanny_values]
            nanny_values = [normalize_text(v) for v in (nanny_values or [])]
            if any(v in nanny_values for v in values):
                match_count += 1

        # Reglas según cantidad de filtros opcionales seleccionados
        num_selected_filters = sum(1 for v in optional_filters.values() if v)
        required_matches = 0
        if num_selected_filters == 1:
            required_matches = 1
        elif num_selected_filters >= 2:
            required_matches = 2
        # Si no hay filtros opcionales seleccionados, required_matches = 0 (cualquier nanny pasa)

        if match_count < required_matches:
            continue

        # --- Scoring manual ---
        score = 2  # zona obligatoria
        for key, values in optional_filters.items():
            if not values:
                continue
            nanny_values = nanny.get(key)
            if isinstance(nanny_values, str):
                nanny_values = [nanny_values]
            nanny_values = [normalize_text(v) for v in (nanny_values or [])]
            score += sum(1.5 for v in values if v in nanny_values)  # cada coincidencia opcional

        # --- Predicción ML ---
        proba = None
        if model:
            try:
                df = pd.DataFrame([{
                    "zone": nanny.get("zone"),
                    "career": nanny.get("career"),
                    "courses": nanny.get("courses", []),
                    "qualities": nanny.get("qualities", [])
                }])
                proba = float(model.predict_proba(df)[0][1])
            except Exception:
                proba = None

        final_score = score * 0.6 + (proba * 0.4 if proba is not None else 0)

        entry = {
            "id": nanny.get("id"),
            "name": nanny.get("name"),
            "zone": nanny.get("zone"),
            "career": nanny.get("career"),
            "courses": nanny.get("courses"),
            "qualities": nanny.get("qualities"),
            "availability": nanny.get("availability"),
            "score": round(score, 2),
            "probability": round(proba, 4) if proba is not None else 0,
            "final_score": round(final_score, 4)
        }

        results.append(entry)

    # --- Ordenar por final_score ---
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    if top_k:
        return results[:top_k]

    return results
