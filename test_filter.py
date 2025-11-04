import pandas as pd
from app.utils.storage import load_dataframe
from app.services.trainer import TrainerService
from app.services.filter_service import FilterService

# Cargar último dataset
df = load_dataframe()
if df is None or df.empty:
    print("No hay dataset cargado. Primero ejecuta /api/nannies con datos.")
    exit()

print(f"Dataset cargado con {len(df)} registros.")

# Inicializar trainer y filterer
trainer = TrainerService()
filterer = FilterService()

# Asegurarte de que el modelo existe y se puede cargar
trainer.train_or_load(df)

# Definir filtros de prueba
filters = {
    "qualities": ["disciplinada", "asertiva"],
    "courses": ["teatro_infantil"],
    "careers": ["astrofisica"],
    "zone": "Guadalajara",
    "availability": True
}

# Aplicar filtrado y scoring
results = filterer.filter_and_score(df, filters)

# Mostrar resultados detallados
print(f"Filtros aplicados: {filters}")
print(f"Se encontraron {results['count']} niñeras:\n")

# Calcular coincidencia real basada en los filtros
for nanny in results["nannies"]:
    coincidencias = 0
    total_criterios = 0

    # Comparar zone
    total_criterios += 1
    if nanny["zone"].strip().lower() == filters["zone"].strip().lower():
        coincidencias += 1

    # Comparar availability
    total_criterios += 1
    if nanny["availability"] == filters["availability"]:
        coincidencias += 1

    # Comparar qualities
    if "qualities" in filters and filters["qualities"]:
        total_criterios += 1
        coincidencias += sum(q in nanny["qualities"] for q in filters["qualities"]) / len(filters["qualities"])

    # Comparar courses
    if "courses" in filters and filters["courses"]:
        total_criterios += 1
        coincidencias += sum(c in nanny["courses"] for c in filters["courses"]) / len(filters["courses"])

    # Comparar careers
    if "careers" in filters and filters["careers"]:
        total_criterios += 1
        coincidencias += sum(ca in nanny["career"] for ca in filters["careers"]) / len(filters["careers"])

    # Calcular porcentaje final
    porcentaje = (coincidencias / total_criterios) * 100

    # Mostrar solo ID y porcentaje
    print(f"ID: {nanny['id']} | Coincidencia real: {porcentaje:.2f}%")
