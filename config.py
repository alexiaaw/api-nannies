import os
from dotenv import load_dotenv

load_dotenv()  # carga .env en el entorno

NANNY_API_KEY = os.getenv("NANNY_API_KEY")
print("🔹 NANNY_API_KEY cargada en Flask:", NANNY_API_KEY)  # 🔹 debug
NANNY_MODEL_PATH = os.getenv("NANNY_MODEL_PATH", "saved_models/nanny_tree.pkl")

# Valores permitidos para zone (normalizados)
ALLOWED_ZONES = {"guadalajara", "zapopan", "tlaquepaque", "tlajomulco", "tonala", "desconocido"}
DATA_CSV_PATH = os.path.join("data", "nannies_latest.csv")
