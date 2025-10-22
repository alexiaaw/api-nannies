import os

MODEL_PATH = os.environ.get("NANNY_MODEL_PATH", "saved_models/nanny_tree.pkl")
API_KEY = os.environ.get("NANNY_API_KEY", "change_this_in_production")

ZONA_METROPOLITANA = {"guadalajara", "zapopan", "tlaquepaque", "tonala", "tlajomulco"}

def generate_label(nanny_row):
    """
    Etiqueta de entrenamiento (0 o 1) para el modelo ML.
    
    Regla de Negocio Actualizada: Una niñera es de 'alta calidad' (Clase 1) si
    tiene el curso 'Primeros Auxilios' Y la cualidad 'Paciente'. 
    """
    
    # Si courses/qualities es None, get() devuelve [] y 'or []' lo garantiza.
    courses = nanny_row.get("courses", []) or []
    qualities = nanny_row.get("qualities", []) or []
    
    norm_courses = [str(c).strip().lower() for c in courses]
    norm_qualities = [str(q).strip().lower() for q in qualities]
    
    # Regla de negocio
    has_pa = "primeros auxilios" in norm_courses
    is_patient = "paciente" in norm_qualities
    
    if has_pa and is_patient:
        return 1  # Clase 1: Alta Calidad (para fines de entrenamiento ML)
    return 0  # Clase 0: Otra