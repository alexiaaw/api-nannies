# utils/zones.py
"""
Mapeo inteligente de códigos postales del Área Metropolitana de Guadalajara (AMG).
Basado en los rangos reales de SEPOMEX, agrupados por municipio.
"""

ZONES_RANGES = {
    "Guadalajara": range(44000, 44999), 
    "Zapopan": range(45000, 45246), 
    "Tonalá": range(45400, 45430), 
    "Tlaquepaque": range(45500, 45640), 
    "Tlajomulco": range(45640, 45680), 
}


def get_zone_by_cp(cp: int) -> str:
    """
    Retorna el nombre del municipio según el código postal proporcionado.
    Si no se encuentra coincidencia (o es inválido), retorna 'Zona desconocida'.
    """
    try:
        cp = int(cp)
    
    except (ValueError, TypeError):
        return "Zona desconocida"

    for municipio, rango in ZONES_RANGES.items():
        if cp in rango:
            return municipio
            
    # Si el CP es válido pero no está en el AMG
    return "Zona desconocida"


# Ejemplo de uso
if __name__ == "__main__":
    ejemplos = [44050, 45123, 45410, 45522, 45670, 99999, "abc"]
    for cp in ejemplos:
        print(f"{cp} → {get_zone_by_cp(cp)}")