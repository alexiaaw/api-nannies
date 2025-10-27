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
    "qualities": ["creativa", "asertiva"],
    "courses": ["primeros_auxilios"],
    "career": ["nutricion"],
    "zone": "guadalajara",
    "availability": True
}

# Aplicar filtrado y scoring
results = filterer.filter_and_score(df, filters)

# Mostrar resultados detallados
print(f"Filtros aplicados: {filters}")
print(f"Se encontraron {results['count']} niñeras:\n")

for nanny in results["nannies"]:
    nid = nanny.get("id")
    name = nanny.get("name") or "Sin nombre"
    score = nanny.get("score", 0) * 100  # porcentaje
    print(f"ID: {nid} | Nombre: {name} | Coincidencia: {score:.2f}%")
