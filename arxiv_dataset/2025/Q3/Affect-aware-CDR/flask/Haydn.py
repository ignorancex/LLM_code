#!/usr/bin/env python3
# coding: utf-8

"""
Flask API server for the HaydnEngine VA RecSys.
Provides an endpoint to recommend paintings based on music preferences.

Author:
- Bereket A. Yilma <bereket.yilma@artaicare.com>
- Luis A. Leiva <luis.leiva@uni.lu>
"""

import logging
from flask import Flask, request, jsonify
from waitress import serve
from flask.Haydn_engine import HaydnEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
eng = HaydnEngine()


@app.route("/retrieval", methods=["POST"])
def retrieval():
    """
    Recommend paintings based on user music preferences.

    Args (via JSON body):
        prefs_audio (dict): Music preferences, e.g., {"21": 5, "55": 4}.
    Args (via query parameters):
        n (int, optional): Number of recommendations to return (default: 3).

    Returns:
        List of recommended painting IDs.
    """
    data = request.json
    n = request.args.get("n", default=3, type=int)

    if not isinstance(data, dict) or "prefs_audio" not in data:
        logging.error("Invalid request: 'prefs_audio' not provided.")
        return jsonify({"error": "'prefs_audio' required"}), 400

    preferences = data["prefs_audio"]
    logging.info(f"Received music preferences: {preferences}")

    # Filter out invalid music IDs (non-numeric keys)
    valid_preferences = {}
    for music_id, rating in preferences.items():
        if not str(music_id).isdigit():  # Check if the key is not a number
            logging.warning(f"Invalid music ID '{music_id}' detected and removed.")
        else:
            valid_preferences[music_id] = rating

    if not valid_preferences:
        logging.error("No valid music preferences provided after filtering.")
        return jsonify({"error": "No valid numeric music IDs provided"}), 400

    logging.info(f"Filtered valid music preferences: {valid_preferences}")

    try:
        recommendations = eng.retrieval(valid_preferences, n=n)
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
    serve(app, host="0.0.0.0", port=5004, threads=10)
