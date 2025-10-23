# routes/search_route.py
from flask import Blueprint, request, jsonify
from model.search import search_nannies
from model.decision_tree import NannyDecisionTree
from utils.preprocessing import normalize_nannies_list
import os
from config import MODEL_PATH

search_bp = Blueprint("search", __name__)

# Cargar modelo global
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = NannyDecisionTree()
        model.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@search_bp.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json() or {}
        nannies = data.get("nannies", [])
        filters = data.get("filters", {})

        # --- Validaciones básicas ---
        if not nannies:
            return jsonify({"error": "nannies list cannot be empty"}), 400
        if not filters.get("zone") or "availability" not in filters:
            return jsonify({"error": "zone and availability are required filters"}), 400

        # Asegurar que los filtros opcionales existan
        optional_keys = ["career", "courses", "qualities"]
        for key in optional_keys:
            if key not in filters:
                filters[key] = []

        # Normalizar la lista de niñeras
        normalized = normalize_nannies_list(nannies)

        # Buscar coincidencias usando el modelo (model puede ser None)
        results = search_nannies(normalized, filters, model=model)

        # Garantizar que cada resultado tenga score, probability y final_score
        for r in results:
            r.setdefault("score", 0)
            r.setdefault("probability", 0)
            r.setdefault("final_score", r["score"] * 0.6 + r.get("probability", 0) * 0.4)

        return jsonify({
            "status": "success",
            "matches": len(results),
            "results": results
        })

    except Exception as e:
        # Manejo de errores para que la API no caiga
        return jsonify({"status": "error", "message": str(e)}), 500
