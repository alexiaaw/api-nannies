# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.decision_tree import NannyDecisionTree
from model.search import search_nannies
import os
import pandas as pd
from dotenv import load_dotenv  # para leer .env
import json

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración
MODEL_PATH = os.environ.get("NANNY_MODEL_PATH", "saved_models/nanny_tree.pkl")
API_KEY = os.environ.get("NANNY_API_KEY", "change_this_in_production")

app = Flask(__name__)
CORS(app)

# Cargar modelo si existe
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = NannyDecisionTree()
        model.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Autenticación por token
@app.before_request
def check_auth():
    token = request.headers.get("Authorization")
    if token is None:
        return jsonify({"error":"missing authorization header"}), 401
    if token != f"Bearer {API_KEY}":
        return jsonify({"error":"invalid token"}), 403

# Home
@app.route("/")
def home():
    return jsonify({"message":"nanny_decision_tree API running"})

# Normalización de columnas multi-etiqueta
def normalize_nannies(nannies_list):
    df = pd.DataFrame(nannies_list)
    for col in ["zone", "career", "courses", "qualities"]:
        if col not in df.columns:
            df[col] = None

    def parse_list(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                v = json.loads(x)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            return [s.strip() for s in x.split(",") if s.strip()]
        return []

    df["courses"] = df["courses"].apply(parse_list)
    df["qualities"] = df["qualities"].apply(parse_list)
    return df

# Entrenar modelo con lista de nannies
@app.route("/train", methods=["POST"])
def train():
    data = request.get_json() or {}
    nannies = data.get("nannies", [])

    if not nannies:
        return jsonify({"error": "nannies list cannot be empty"}), 400

    try:
        df = normalize_nannies(nannies)
        feature_cols = ["zone", "career", "courses", "qualities"]
        X = df[feature_cols]

        # y dummy para entrenamiento
        y = pd.Series([1] * len(df))

        global model
        model = NannyDecisionTree()
        model.fit(X, y)
        model.save(MODEL_PATH)
        return jsonify({"status": "success", "trained_nannies": len(nannies)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predicción de nannies según filtros
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    if model is None:
        return jsonify({"error": "model not loaded or failed to initialize"}), 500

    nannies = data.get("nannies", [])
    if not nannies:
        return jsonify({"error": "nannies list cannot be empty"}), 400

    # Normalizar nannies antes de predecir
    df = normalize_nannies(nannies)
    feature_cols = ["zone", "career", "courses", "qualities"]
    X = df[feature_cols]

    # Filtros
    career = data.get("career", [])
    if isinstance(career, str):
        career = [career]

    courses = data.get("courses", [])
    if isinstance(courses, str):
        courses = [courses]

    qualities = data.get("qualities", [])
    if isinstance(qualities, str):
        qualities = [qualities]

    postal_code = data.get("postal_code", "")
    zone = data.get("zone", "")

    filters = {
        "career": career,
        "courses": courses,
        "qualities": qualities,
        "postal_code": postal_code,
        "zone": zone,
        "availability": True
    }

    results = search_nannies(nannies, filters, model=model)

    return jsonify({"status":"success", "matches":len(results), "results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
