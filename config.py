# config.py
import os
from dotenv import load_dotenv

# Variables desde el archivo .env 
load_dotenv()


MODEL_PATH = os.environ.get("NANNY_MODEL_PATH", "saved_models/nanny_tree.pkl")
API_KEY = os.environ.get("NANNY_API_KEY", "change_this_in_production")


ZONA_METROPOLITANA = {"guadalajara", "zapopan", "tlaquepaque", "tonala", "tlajomulco"}

# === Generación de etiqueta (para entrenamiento del modelo ML) ===
def generate_label(nanny_row):
    """
    Genera la etiqueta de entrenamiento (0 o 1) para el modelo ML.
    
    """

    # Asegurar que los campos sean listas
    courses = nanny_row.get("courses", []) or []
    qualities = nanny_row.get("qualities", []) or []
    careers = nanny_row.get("career", []) or []

    # Normalización
    norm_courses = [str(c).strip().lower() for c in courses]
    norm_qualities = [str(q).strip().lower() for q in qualities]
    norm_careers = (
        [str(c).strip().lower() for c in careers]
        if isinstance(careers, list)
        else [str(careers).strip().lower()]
    )

    # Reglas de negocio
    has_pa = "primeros auxilios" in norm_courses
    is_patient = "paciente" in norm_qualities
    related_career = any(
        c in norm_careers
        for c in ["educación", "pedagogía", "psicología", "psicopedagogía"]
    )

    # Condición final para clase 1 (Alta calidad)
    if (has_pa and is_patient) or related_career:
        return 1
    return 0
