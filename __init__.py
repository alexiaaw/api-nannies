# api/__init__.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from config import API_KEY
from routes.search_route import search_bp
from routes.train_route import train_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    # --- Autenticación global ---
    @app.before_request
    def check_auth():
        if request.endpoint == 'home':
            return  # la raíz no requiere token
        token = request.headers.get("Authorization")
        if token is None:
            return jsonify({"error": "missing authorization header"}), 401
        if token != f"Bearer {API_KEY}":
            return jsonify({"error": "invalid token"}), 403

    # --- Registrar rutas ---
    app.register_blueprint(search_bp)
    app.register_blueprint(train_bp)

    # --- Ruta raíz ---
    @app.route("/")
    def home():
        return jsonify({"message": "nanny_decision_tree API running"})

    return app
