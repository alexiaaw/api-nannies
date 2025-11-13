import os
from flask import Flask, jsonify
from .routes.nannies import nannies_bp
from dotenv import load_dotenv

def create_app():
    load_dotenv()  # carga .env en el entorno

    app = Flask(__name__)

    # Asignar API key y ruta del modelo a app.config
    app.config['NANNY_API_KEY'] = os.getenv("NANNY_API_KEY")
    app.config['NANNY_MODEL_PATH'] = os.getenv("NANNY_MODEL_PATH", "saved_models/nanny_tree.pkl")

    # registrar blueprints
    app.register_blueprint(nannies_bp, url_prefix="/api")

     #ruta raíz para comprobar que la API está viva
    @app.route("/")
    def home():
        return jsonify({"message": "API Nannies is running!"})

    # errores comunes
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({"error": "Unauthorized", "message": str(e)}), 401

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "message": str(e)}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

    return app
