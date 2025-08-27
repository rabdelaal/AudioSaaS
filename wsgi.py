from app import app

if __name__ == "__main__":
    # For production use a WSGI server like gunicorn or waitress
    app.run(host="0.0.0.0", port=5000)
