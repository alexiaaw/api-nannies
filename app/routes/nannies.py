"""
Routes para nannies:
- POST /api/nannies        : Recibe dataset de Laravel, valida, guarda DataFrame, entrena modelo (o carga si existe).
- POST /api/nannies/filter : Recibe filtros opcionales, filtra DataFrame y usa el modelo para puntuar coincidencias.
"""

from flask import Blueprint, request, jsonify, current_app
from ..utils.auth import require_api_key
from ..utils.validation import validate_nannies_payload, validate_filter_payload
from ..utils.storage import save_dataframe, load_dataframe
from ..services.trainer import TrainerService
from ..services.filter_service import FilterService
import traceback

nannies_bp = Blueprint("nannies", __name__)

trainer = TrainerService()
filterer = FilterService()


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
    except Exception as e:
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "server_error", "message": "Error procesando dataset."}), 500


@nannies_bp.route("/nannies/filter", methods=["POST"])
@require_api_key
def filter_nannies():
    """
    Endpoint para filtrar niñeras usando filtros opcionales y puntuar resultados.
    """
    try:
        payload = request.get_json(force=True)
        filters = validate_filter_payload(payload)  # Puede estar vacío

        # Cargar último dataset
        df = load_dataframe()
        if df is None or df.empty:
            return jsonify({
                "error": "no_data",
                "message": "No hay dataset cargado. Enviar datos a /api/nannies primero."
            }), 400

        # -----------------
        # FILTROS OPCIONALES
        # -----------------
        # Zona
        if filters.get("zone"):
            zone = filters["zone"].strip().lower()
            df = df[df["zone"].str.lower() == zone]

        # Disponibilidad
        availability = filters.get("availability", True)  # default True
        df = df[df["availability"] == availability]

        # Qualities
        if filters.get("qualities"):
            qualities = set(filters["qualities"])
            df = df[df["qualities"].apply(lambda q_list: qualities.issubset(set(q_list)))]

        # Courses
        if filters.get("courses"):
            courses = set(filters["courses"])
            df = df[df["courses"].apply(lambda c_list: courses.issubset(set(c_list)))]

        # Career
        if filters.get("career"):
            careers = set(filters["career"])
            df = df[df["career"].apply(lambda cr_list: careers.issubset(set(cr_list)))]

        # -----------------
        # SCORING CON MODELO
        # -----------------
        results = filterer.filter_and_score(df, filters)

        if not results:
            current_app.logger.info("No se encontraron coincidencias para los filtros: %s", filters)

        return jsonify({
            "count": len(results),
            "filters_applied": filters,
            "results": results
        }), 200

    except ValueError as e:
        return jsonify({"error": "validation_error", "message": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": "no_data", "message": str(e)}), 400
    except Exception as e:
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "server_error", "message": "Error filtrando niñeras."}), 500
