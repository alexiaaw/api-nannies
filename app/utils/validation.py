from typing import Dict, Any
import pandas as pd
import config

# Allowed values según enums de PHP
ALLOWED_ZONES = set([
    'desconocido', 'guadalajara', 'zapopan', 'tonala', 'tlaquepaque', 'tlajomulco'
])

ALLOWED_QUALITIES = set([
    'empatica', 'creativa', 'paciente', 'carinosa', 'observadora', 
    'asertiva', 'proactiva', 'flexible', 'ludica', 'bilingue'
])

ALLOWED_COURSES = set([
    'primeros_auxilios', 'cuidado_infantil', 'desarrollo_infantil',
    'nutricion_y_alimentacion', 'educacion_y_aprendizaje', 'psicologia_infantil',
    'disciplina_y_comportamiento', 'lactancia_y_cuidado_bebes',
    'inclusion_y_diversidad', 'comunicacion_y_lenguaje'
])

ALLOWED_CAREERS = set([
    'pedagogia', 'psicologia', 'enfermeria', 'docencia', 'nutricion',
    'trabajo_social', 'psicopedagogia', 'terapia_psicomotriz', 'pediatria',
    'artes_escenicas_danza'
])

# Mapeo de zonas para normalizar variantes
ZONE_MAPPING = {z: z for z in ALLOWED_ZONES}
ZONE_MAPPING.update({
    'desconocida': 'desconocido',
    'desconocido': 'desconocido'
})

def _normalize_zone(z: str) -> str:
    if not z:
        print("[WARNING] Zone missing or empty, defaulting to 'desconocido'")
        return 'desconocido'
    z_norm = str(z).strip().lower()
    if z_norm not in ZONE_MAPPING:
        print(f"[WARNING] Unknown zone '{z}', defaulting to 'desconocido'")
        return 'desconocido'
    return ZONE_MAPPING[z_norm]

def _normalize_list(values: list, allowed_set: set, field_name: str) -> list:
    if not values:
        return []
    normalized = []
    for v in values:
        v_norm = str(v).strip().lower()
        if v_norm in allowed_set:
            normalized.append(v_norm)
        else:
            print(f"[WARNING] Ignoring invalid {field_name} value: '{v}'")
    return normalized

def validate_nannies_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    if "nannies" not in payload:
        raise ValueError("Missing 'nannies' field.")
    nannies = payload["nannies"]
    if not isinstance(nannies, list):
        raise ValueError("'nannies' must be a list.")

    normalized = []
    for item in nannies:
        if not isinstance(item, dict):
            raise ValueError("Each nanny must be an object.")
        if "availability" not in item or "zone" not in item:
            raise ValueError(f"Nanny entry missing required fields: {item}")

        zone = _normalize_zone(item.get("zone"))
        qualities = _normalize_list(item.get("qualities"), ALLOWED_QUALITIES, "quality")
        courses = _normalize_list(item.get("courses"), ALLOWED_COURSES, "course")
        careers = _normalize_list(item.get("career"), ALLOWED_CAREERS, "career")

        normalized.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "qualities": qualities,
            "courses": courses,
            "career": careers,
            "zone": zone,
            "availability": bool(item.get("availability"))
        })

    return pd.DataFrame(normalized)

def validate_filter_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    if "zone" not in payload or "availability" not in payload:
        raise ValueError("Filter missing 'zone' or 'availability' (obligatorio).")

    zone = _normalize_zone(payload.get("zone"))
    availability = bool(payload.get("availability"))
    qualities = _normalize_list(payload.get("qualities"), ALLOWED_QUALITIES, "quality")
    courses = _normalize_list(payload.get("courses"), ALLOWED_COURSES, "course")
    careers = _normalize_list(payload.get("career"), ALLOWED_CAREERS, "career")

    return {
        "zone": zone,
        "availability": availability,
        "qualities": qualities,
        "courses": courses,
        "career": careers
    }
