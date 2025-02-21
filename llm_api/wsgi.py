from llm_api import app
import os

if __name__ == "__main__":
    set_debug = os.environ.get("FLASK_ENV", "production") == "development"
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 80))

    app.run(
        host=host,
        port=port,
        debug=set_debug,
    )
