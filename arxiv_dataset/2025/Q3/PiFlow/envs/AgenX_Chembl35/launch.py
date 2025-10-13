#!/usr/bin/env python
"""
ChEMBL prediction module launcher.
This module can be used standalone or as part of the unified server.
"""
import os
import sys
import logging
from flask import Flask, request, jsonify

# Add parent directory to path for unified server imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.AgenX_Chembl35.inference import predict_smiles

# Module configuration
MODULE_NAME = "ChEMBL"
MODEL_PATH = os.path.join("models", "best_r2_model.pt")
HOST = "127.0.0.1"
PORT = 12500


def create_chembl_app():
    """Create Flask app for ChEMBL prediction API"""
    app = Flask(__name__)

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = app.logger

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        API endpoint for pChEMBL prediction from SMILES.

        Expected JSON input:
        {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O"
        }

        Returns:
        {
            "input": "CC(=O)Oc1ccccc1C(=O)O",
            "output": 5.23,
            "success": true,
            "error": ""
        }
        """
        response = {"input": "", "output": "", "success": False, "error": ""}

        try:
            # Get JSON data from request
            data = request.get_json()
            if not data:
                response["error"] = "No JSON data provided"
                return jsonify(response), 400

            # Extract SMILES
            smiles = data.get("smiles")
            logger.debug(f"Received SMILES: {smiles}")

            if not smiles:
                response["error"] = "SMILES string is missing in request body"
                return jsonify(response), 400

            # Make prediction
            predicted = predict_smiles(smiles, model_path=MODEL_PATH)
            logger.debug(f"Predicted value: {predicted}")

            if predicted is not None:
                response["input"] = smiles
                response["output"] = float(predicted)
                response["success"] = True
                return jsonify(response), 200
            else:
                response["input"] = smiles
                response["error"] = "Invalid SMILES string or prediction failed"
                return jsonify(response), 400

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            response["error"] = f"Internal server error: {str(e)}"
            return jsonify(response), 500

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify(
            {
                "status": "healthy",
                "module": MODULE_NAME,
                "model_available": os.path.exists(MODEL_PATH),
            }
        )

    return app


def main():
    """Run standalone ChEMBL server"""
    print(f"Starting {MODULE_NAME} prediction server...")
    print(f"Available at: http://{HOST}:{PORT}")
    print("Endpoints:")
    print("  POST /predict - ChEMBL prediction")
    print("  GET /health - Health check")
    print("\nNote: For unified server with all models, use envs/unified_server.py")

    app = create_chembl_app()
    app.run(debug=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
