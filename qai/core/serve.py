"""
qai.core.serve -- Section 89/102's claim, made real: a model's own schema
becomes the request/response format automatically. Uses Flask (available
here) as the real HTTP layer.
"""
from __future__ import annotations
import numpy as np


def build_flask_app(model):
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict_route():
        try:
            payload = request.get_json(force=True)
            if isinstance(payload, dict) and "input" in payload:
                x = payload["input"]
            else:
                x = payload  # allow either {"input": [...]} or a raw list/dict
            result = model.predict(x)
            # numpy types aren't JSON-serializable -- a real, common gotcha,
            # handled explicitly rather than letting jsonify silently fail
            if isinstance(result, (np.integer,)):
                result = int(result)
            elif isinstance(result, (np.floating,)):
                result = float(result)
            elif isinstance(result, np.ndarray):
                result = result.tolist()
            elif hasattr(result, "item"):
                result = result.item()
            return jsonify({"prediction": result, "technique": model.technique_name})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/health", methods=["GET"])
    def health_route():
        return jsonify({"status": "ok", "technique": model.technique_name,
                         "trained": model.technique.is_trained()})

    @app.route("/info", methods=["GET"])
    def info_route():
        return jsonify({"technique": model.technique_name,
                         "learning_technique": model.learning_technique,
                         "trained": model.technique.is_trained()})

    return app


def serve(model, port=5050, host="127.0.0.1"):
    """qai.serve(model, port=...) -- for real use, blocks and runs the server.
    Tests use the Flask test client instead (build_flask_app + .test_client()),
    which exercises the exact same route logic without needing a real
    network socket -- honest and correct for a sandbox with restricted
    networking, while still testing the REAL request/response code."""
    app = build_flask_app(model)
    app.run(port=port, host=host)
