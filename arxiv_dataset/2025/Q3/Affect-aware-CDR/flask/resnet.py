#!/usr/bin/env python3
# coding: utf-8

"""
Flask API server for the ResNetEngine RecSys.
Provides an endpoint to recommend paintings based on painting preferences.

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
- Luis A. Leiva <name.surname@uni.lu>
"""

import logging
from flask import Flask, request, jsonify
from waitress import serve
from resnet_engine import ResNetEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
eng = ResNetEngine()


@app.route("/retrieval", methods=["POST"])
def retrieval():
    """
    Recommend paintings based on user painting preferences.

    Args (via JSON body):
        prefs_image (dict): Painting preferences, e.g., {"57726e48edc2cb3880b69b14": 5}.
    Args (via query parameters):
        n (int, optional): Number of recommendations to return (default: 3).

    Returns:
        List of recommended painting IDs.
    """
    data = request.json
    n = request.args.get("n", default=3, type=int)

    if not isinstance(data, dict) or "prefs_image" not in data:
        logging.error("Invalid request: 'prefs_image' not provided.")
        return jsonify({"error": "'prefs_image' required"}), 400

    preferences = data["prefs_image"]
    logging.info(f"Received painting preferences: {preferences}")

    try:
        recommendations = eng.retrieval(preferences, n=n)
        logging.info(f"Recommendations generated: {recommendations}")
        return jsonify(recommendations)
    except ValueError as e:
        logging.error(f"Retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Use a production server instead of Flask's built-in one
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5006, threads=10)
