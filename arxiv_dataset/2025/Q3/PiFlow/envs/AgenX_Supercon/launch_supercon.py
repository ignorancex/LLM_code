#!/usr/bin/env python
"""
Superconductor critical temperature prediction module launcher.
This module can be used standalone or as part of the unified server.
"""
import os
import sys
import logging
from flask import Flask, request, jsonify
import torch
import pickle
import pandas as pd

# Add parent directory to path for unified server imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TcPredictor
from src.data_processor import SuperconDataProcessor

# Module configuration
MODULE_NAME = "Superconductor"
HOST = "127.0.0.1"
PORT = 12502
MODEL_PATH = "./models/best_supercon_model.pth"
PROCESSOR_PATH = "./models/supercon_processor.pkl"

# Tutorial documentation
TUTORIAL_DOCUMENT = """
# Superconductor Critical Temperature (Tc) Prediction Tool

## About Superconductivity:
- Superconductors are materials that conduct electricity with zero resistance below a critical temperature (Tc).
- Higher Tc values are desirable for practical applications, as they require less cooling.
- Finding new superconductors with higher Tc values is a major goal in materials science.

## Input Parameters:
- element: Chemical formula of the material (e.g., "Ba0.2La1.8Cu1O4-Y")
- str3 (optional): Structure type code (e.g., "T" for tetragonal)

## Prediction Output:
- The tool returns the predicted critical temperature (Tc) in Kelvin.

## Example Usage:
```json
{
  "element": "Ba0.2La1.8Cu1O4-Y",
  "str3": "T"
}
```

## Notes:
- Chemical formulas should follow standard notation with element symbols and numeric proportions
- Common structure types include:
  - T: Tetragonal
  - C: Cubic
  - H: Hexagonal
  - O: Orthorhombic
  - M: Monoclinic

This tool enables rapid virtual experimentation with superconducting materials, allowing the exploration of compositional space without the need for expensive and time-consuming physical experiments.
"""


def load_supercon_model():
    """Load the trained model and get input size from saved processor data"""
    try:
        # Get input size from processor data
        with open(PROCESSOR_PATH, "rb") as f:
            processor_data = pickle.load(f)
            input_size = processor_data["n_features"]

        # Load model with correct input size
        model = TcPredictor(input_size=input_size)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        return model, input_size
    except Exception as e:
        return None, None


def create_supercon_app():
    """Create Flask app for Superconductor prediction API"""
    app = Flask(__name__)
    app.logger.setLevel(logging.DEBUG)

    # Load model and processor
    try:
        model, input_size = load_supercon_model()
        processor = SuperconDataProcessor()
        app.logger.info(f"Model loaded successfully with input size: {input_size}")
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        model = None
        input_size = None
        processor = None

    @app.route("/predict", methods=["POST"])
    def predict_tc():
        submit = {"input": "", "output": "", "success": False, "error": ""}

        try:
            # Check if model is loaded
            if model is None:
                submit["error"] = "Model not loaded. Please check server logs."
                return jsonify(submit), 500

            # Get input data from request
            data = request.get_json()
            app.logger.debug(f"Received input data: {data}")

            if not data:
                submit["error"] = "Input data is missing in request body"
                return jsonify(submit), 400  # Bad Request

            # Check for required fields
            if "element" not in data:
                submit["error"] = "Missing required field: 'element' (chemical formula)"
                return jsonify(submit), 400

            # Store input data
            submit["input"] = data

            # Process input data
            input_df = pd.DataFrame([data])
            try:
                processed_input = processor.process_input(input_df)
                input_tensor = torch.FloatTensor(processed_input)
            except Exception as e:
                app.logger.error(f"Error processing input: {str(e)}")
                submit["error"] = f"Error processing input: {str(e)}"
                return jsonify(submit), 400

            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_tc = float(prediction[0][0])

            app.logger.debug(f"Predicted Tc: {predicted_tc}")

            # Update response with prediction
            submit["output"] = predicted_tc
            submit["success"] = True
            return jsonify(submit)

        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            submit["error"] = f"Prediction failed: {str(e)}"
            return jsonify(submit), 500  # Internal Server Error

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        models_available = (
            model is not None
            and os.path.exists(MODEL_PATH)
            and os.path.exists(PROCESSOR_PATH)
        )

        return jsonify(
            {
                "status": "healthy",
                "module": MODULE_NAME,
                "model_available": models_available,
            }
        )

    @app.route("/tutorial", methods=["GET"])
    def tutorial():
        """Return tutorial documentation"""
        return jsonify({"tutorial": TUTORIAL_DOCUMENT})

    return app


def main():
    """Run standalone Superconductor server"""
    print(f"Starting {MODULE_NAME} prediction server...")
    print(f"Available at: http://{HOST}:{PORT}")
    print("Endpoints:")
    print("  POST /predict - Superconductor Tc prediction")
    print("  GET /health - Health check")
    print("  GET /tutorial - Tutorial documentation")
    print("\nNote: For unified server with all models, use envs/unified_server.py")

    app = create_supercon_app()
    app.run(debug=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
