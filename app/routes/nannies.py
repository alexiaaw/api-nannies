"""
Routes para nannies:
- POST /api/nannies        : Recibe dataset de Laravel, valida, guarda DataFrame, entrena modelo (o carga si existe).
- POST /api/nannies/filter : Recibe filtros opcionales y devuelve SIEMPRE 3 IDs de niñeras (exactas o similares).
"""

from flask import Blueprint, request, jsonify, current_app
from ..utils.auth import require_api_key
from ..utils.validation import validate_nannies_payload, validate_filter_payload
from ..utils.storage import save_dataframe, load_dataframe
from ..services.trainer import TrainerService
import traceback
import pandas as pd
import random

nannies_bp = Blueprint("nannies", __name__)

trainer = TrainerService()

# Zonas válidas según tu enum en Laravel
VALID_ZONES = {"guadalajara", "zapopan", "tlaquepaque", "tlajomulco", "tonala"}


@nannies_bp.route("/nannies", methods=["POST"])
@require_api_key
def receive_nannies():
    """
    Endpoint para recibir dataset desde Laravel y entrenar o cargar el modelo.
    """
    current_app.logger.info("HEADERS RECIBIDOS: %s", dict(request.headers))
    try:
        payload = request.get_json(force=True)
        current_app.logger.debug("Datos recibidos: %s", payload)

        # Validación y transformación a DataFrame
        df = validate_nannies_payload(payload)

        # Guardar último dataset
        save_dataframe(df)

        # Entrenar o cargar modelo
        trainer.train_or_load(df)

        return jsonify({
            "message": "Dataset recibido y modelo entrenado/actualizado",
            "count": len(df)
        }), 200

    except ValueError as e:
        return jsonify({"error": "validation_error", "message": str(e)}), 400
    except Exception:
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "server_error", "message": "Error procesando dataset."}), 500


@nannies_bp.route("/nannies/filter", methods=["POST"])
@require_api_key
def filter_nannies():
    """
    Endpoint para filtrar niñeras.
    Devuelve SIEMPRE 3 niñeras (exactas o similares), 
    priorizando coincidencias completas, pero manteniendo la zona y disponibilidad como obligatorias.
    """
    try:
        payload = request.get_json(force=True)
        current_app.logger.info(">>> PAYLOAD RECIBIDO EN /filter: %s", payload)
        filters = validate_filter_payload(payload)

        # Cargar último dataset
        df = load_dataframe()
        if df is None or df.empty:
            return jsonify({
                "error": "no_data",
                "message": "No hay dataset cargado. Enviar datos a /api/nannies primero."
            }), 400

        # -----------------
        # NORMALIZAR ZONA
        # -----------------
        zone = filters.get("zone", "").strip().lower()
        if zone not in VALID_ZONES:
            return jsonify({
                "error": "invalid_zone",
                "message": f"Zona '{zone}' no reconocida. Debe ser una de: {', '.join(VALID_ZONES)}"
            }), 400

        # Filtrar por zona y disponibilidad
        df = df[df["zone"].str.lower().str.strip() == zone]
        df = df[df["availability"] == filters.get("availability", True)]

        if df.empty:
            return jsonify({
                "error": "no_matches",
                "message": "No hay niñeras disponibles en esa zona con esa disponibilidad."
            }), 200

        # -----------------
        # FILTROS EXACTOS
        # -----------------
        df_exact = df.copy()

        if filters.get("qualities"):
            qualities = set(filters["qualities"])
            df_exact = df_exact[df_exact["qualities"].apply(lambda q: qualities.issubset(set(q)))]

        if filters.get("courses"):
            courses = set(filters["courses"])
            df_exact = df_exact[df_exact["courses"].apply(lambda c: courses.issubset(set(c)))]

        if filters.get("career"):
            careers = set(filters["career"])
            df_exact = df_exact[df_exact["career"].apply(lambda cr: careers.issubset(set(cr)))]

        # -----------------
        # SI HAY 3 O MÁS EXACTAS → usar esas
        # -----------------
        if len(df_exact) >= 3:
            df_final = df_exact.sample(n=3, random_state=42)

        else:
            # Menos de 3 exactas: buscar similares
            df_final = df_exact.copy()
            
            
            # Evitar error si df_final no tiene columna 'id'
            if "id" not in df_final.columns:
                df_final["id"] = []

            if "id" not in df.columns:
                df["id"] = range(len(df))

            # Ahora sí, podemos usar la operación segura
            remaining = df[~df["id"].isin(df_final["id"])]


            # Similitud basada en cantidad de coincidencias parciales
            def similarity(row):
                score = 0
                if filters.get("qualities"):
                    score += len(set(filters["qualities"]).intersection(set(row["qualities"])))
                if filters.get("courses"):
                    score += len(set(filters["courses"]).intersection(set(row["courses"])))
                if filters.get("career"):
                    score += len(set(filters["career"]).intersection(set(row["career"])))
                return score

            remaining["similarity"] = remaining.apply(similarity, axis=1)
            remaining = remaining.sort_values(by="similarity", ascending=False)

            # Tomar las más similares hasta completar 3
            faltan = 3 - len(df_final)
            if faltan > 0 and not remaining.empty:
                extra = remaining.head(faltan)
                df_final = pd.concat([df_final, extra])

        # -----------------
        # RESPUESTA FINAL (solo IDs)
        # -----------------
        nanny_ids = df_final["id"].tolist()
        random.shuffle(nanny_ids)  # para variar el orden sin afectar el resultado

        current_app.logger.info(">>> IDs filtrados enviados a Laravel: %s", nanny_ids)
        
        # Convertimos los IDs a objetos
        top3_nannies = [{"id": id} for id in nanny_ids]
        
        return jsonify({
            "filters_applied": filters,
            "count": len(nanny_ids),
            "top3Nannies": top3_nannies
        }), 200

    except ValueError as e:
        return jsonify({"error": "validation_error", "message": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": "no_data", "message": str(e)}), 400
    except Exception:
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "server_error", "message": "Error filtrando niñeras."}), 500
