# utils/preprocessing.py

def normalize_list_field(field):
    """Convierte cualquier campo en lista de strings"""
    if not field:
        return []
    if isinstance(field, list):
        return [str(f).strip() for f in field]
    return [str(field).strip()]

def normalize_zone(zone):
    """Normaliza zona a minúsculas y sin espacios extra"""
    if not zone:
        return ""
    return str(zone).strip().lower()

def normalize_availability(avail):
    """Normaliza disponibilidad, si existe; si no, devuelve lista vacía"""
    if not avail:
        return []
    if isinstance(avail, list):
        return [str(a).strip() for a in avail]
    return [str(avail).strip()]

def normalize_nannies_list(nannies):
    """
    Recibe una lista de diccionarios (nannies) y normaliza todos los campos relevantes.
    Retorna una lista lista para ser procesada por el modelo ML o buscador.
    """
    normalized = []
    for n in nannies:
        try:
            nn = dict(n)  # copiar diccionario

            # Normalizar listas
            nn["courses"] = normalize_list_field(nn.get("courses"))
            nn["qualities"] = normalize_list_field(nn.get("qualities"))
            nn["career"] = normalize_list_field(nn.get("career"))

            # Normalizar disponibilidad
            nn["availability"] = normalize_availability(nn.get("availability"))

            # Normalizar zonas
            zone_val = nn.get("zone") or nn.get("zona") or nn.get("ubicacion")
            if isinstance(zone_val, list):
                nn["zone"] = [normalize_zone(z) for z in zone_val]
            else:
                nn["zone"] = [normalize_zone(zone_val)]

            # Normalizar nombre
            if "name" in nn:
                nn["name"] = str(nn["name"]).strip()

            # Asegurar todas las columnas esperadas
            for col in ["zone", "career", "courses", "qualities"]:
                if col not in nn:
                    nn[col] = []

            normalized.append(nn)

        except Exception as e:
            print(f"Error normalizando nanny {n.get('id', 'unknown')}: {e}")
            continue

    return normalized
