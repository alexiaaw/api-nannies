# routes/train_route.py
from flask import Blueprint, request, jsonify
from model.decision_tree import NannyDecisionTree
from utils.preprocessing import normalize_nannies_list
import pandas as pd
from config import MODEL_PATH, generate_label

train_bp = Blueprint("train", __name__)

# Modelo global
model = None

@train_bp.route("/train", methods=["POST"])
def train():
    global model
    data = request.get_json() or {}
    nannies = data.get("nannies", [])

    if not nannies:
        return jsonify({"error": "nannies list cannot be empty"}), 400

    # Normalizar datos
    normalized_list = normalize_nannies_list(nannies)
    df = pd.DataFrame(normalized_list)

    # Convertir columnas que pueden ser listas en strings
    for col in ["career", "courses", "qualities"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ",".join(x) if isinstance(x, list) else str(x))

    # Columnas que el modelo espera
    feature_cols = ["zone", "career", "courses", "qualities"]

    # Asegurar que existan todas las columnas
    for col in feature_cols:
        if col not in df.columns:
            df[col] = [""] * len(df)

    # ✅ Asegurar que X sea un DataFrame con nombres de columna correctos
    X = pd.DataFrame(df[feature_cols].copy(), columns=feature_cols)

    # Generar etiquetas reales
    y = df.apply(generate_label, axis=1)

    # Debug para ver qué llega del lado de Laravel
    print("Tipo de X:", type(X))
    print("Columnas de X:", X.columns.tolist())
    print("Primeras filas de X:\n", X.head())
    print("Primeras filas de y:\n", y.head())

    # Entrenar y guardar modelo
    model = NannyDecisionTree()
    model.fit(X, y)
    model.save(MODEL_PATH)

    return jsonify({
        "status": "success",
        "trained_nannies": len(nannies),
        "message": "Modelo entrenado usando etiquetas reales de alta calidad"
    })
