from utils.zones import get_zone_by_cp
from utils.preprocessing import normalize_nannies_list
import pandas as pd


def normalize_text(s):
    return str(s).strip().lower() if s is not None else ""

def zone_matches(nanny_zone, user_zone):
    """
    Verifica si la zona de la niñera coincide con la zona del usuario.
    Retorna True si hay coincidencia. Retorna False si la zona del usuario es inválida.
    """
    n = normalize_text(nanny_zone)
    u = normalize_text(user_zone)
    
    # Si la zona del usuario es inválida, devuelve False para que no sume puntos.
    if u == "código postal inválido" or u == "zona desconocida":
        return False
    
    if "zona metropolitana" in n or "metropolitana" in n:
        return True

    return u in n or n in u or u == n

def search_nannies(nannies_list, filters, model=None, require_all_multi=True, top_k=None):
    """
    Filtra, puntúa y clasifica a las niñeras basándose en filtros y un modelo ML.
    La lista devuelta es 100% compatible con todos los filtros proporcionados.
    """
    nannies = normalize_nannies_list(nannies_list)
    
    # Extracción y Normalización de Filtros.
    career_f = filters.get("career") or []
    if isinstance(career_f, str):
        career_f = [career_f]  # Convertir string a lista para unificar formato
    courses_f = filters.get("courses") or []
    qualities_f = filters.get("qualities") or []
    availability_filter = filters.get("availability", None)
    address = filters.get("address", {}) or {}
    postal_code = address.get("postal_code") or address.get("postal")
    user_zone = get_zone_by_cp(postal_code)
    
    av_text = str(availability_filter).strip().lower()
    require_available = av_text in ["disponible", "true", "1", "sí", "si"] or availability_filter is True

    results = []
    
    for nanny in nannies:
        

        # Filtros obligatorios.
        if require_available and not nanny.get("availability", False):
            continue

        if not zone_matches(nanny.get("zone",""), user_zone):
            continue

        if career_f:
            nanny_career = normalize_text(nanny.get("career", ""))
            if not any(nanny_career == normalize_text(c) for c in career_f):
                 continue

        
        if courses_f:
            nanny_courses = [normalize_text(c) for c in nanny.get("courses", [])]
            # Usamos la lógica estricta require_all_multi=True para el filtrado
            if not all(normalize_text(c) in nanny_courses for c in courses_f):
                continue
        
        if qualities_f:
            nanny_quals = [normalize_text(q) for q in nanny.get("qualities", [])]
            # Usamos la lógica estricta require_all_multi=True para el filtrado
            if not all(normalize_text(q) in nanny_quals for q in qualities_f):
                continue
                
        
        # SCORING Y RANKING (Solo para niñeras que pasaron todos los filtros)
        
        score = 0
        
        # ZONA 
        score += 2

        if career_f:
            score += 2

        if courses_f:
            score += len(courses_f) * 1.5
            
        if qualities_f:
            score += len(qualities_f) * 1.2
            
        # PREDICCIÓN DEL MODELO ML
        proba = None
        if model:
            # Lógica para crear DataFrame y obtener proba (se mantiene)
            df = pd.DataFrame([{
                "zone": nanny.get("zone"),
                "career": nanny.get("career"),
                "courses": nanny.get("courses", []),
                "qualities": nanny.get("qualities", [])
            }])
            try:
                proba = float(model.predict_proba(df)[0][1])
            except Exception:
                proba = None

        # FILTRO FINAL Y CÁLCULO DE PUNTUACIÓN (Se mantiene el filtro score <= 0)
        if score <= 0 and not proba:
            continue

        entry = {
            "id": nanny.get("id"),
            # ... (Resto de la información de la niñera)
            "final_score": round(score * 0.6 + (proba * 0.4 if proba is not None else 0), 4),
            "score": round(score, 2)
        }
        if proba is not None:
             entry["probability"] = round(proba, 4)

        results.append(entry)

    # Ordenamiento y Límite
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    if top_k:
        return results[:top_k]
    return results