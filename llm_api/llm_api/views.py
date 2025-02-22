from llm_api import app
from flask import jsonify, request
from llm_api.ollama import new_ollama_instance
from llm_api.analyze import analyze_inputs


@app.route("/", methods=["GET"])
def index():
    return "Hello World!"


@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    required_fields = ["job_text", "total_job_info", "session_id"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return (
            jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}),
            400,
        )

    ollama = new_ollama_instance()

    result, job_retriever, resume_retriever = analyze_inputs(
        data["job_text"], data["total_job_info"], data["session_id"], ollama
    )

    return jsonify({"result": result}), 200


@app.route("/api/v1/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    if "user_input" not in data:
        return jsonify({"error": "Missing user_input"}), 400

    return jsonify({"message": "JSON data received", "data": data}), 200
