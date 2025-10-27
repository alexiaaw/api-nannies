from app import create_app

app = create_app()

if __name__ == "__main__":
    # Para desarrollo; en producción usar un WSGI (gunicorn/uwsgi)
    app.run(host="0.0.0.0", port=5000, debug=True)
