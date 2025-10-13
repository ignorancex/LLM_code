#!/usr/bin/env python
"""
Unified prediction server for PiEvo AgentX modules.
Serves prediction endpoints for ChEMBL, Nanohelix, and Superconductor models.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import joblib
import pickle

# Add envs modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import prediction functions
from AgenX_Chembl35.inference import predict_smiles
from AgenX_Supercon.src.model import TcPredictor
from AgenX_Supercon.src.data_processor import SuperconDataProcessor

# Flask app setup
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# Configuration
HOST = "127.0.0.1"
PORT = 12500

# Model paths
CHEMBL_MODEL_PATH = os.path.join("AgenX_Chembl35", "models", "best_r2_model.pt")
NANOHELIX_MODEL_PATH = os.path.join(
    "AgenX_Nanohelix", "models", "nanohelix_mlp_model.pkl"
)
NANOHELIX_SCALER_X_PATH = os.path.join(
    "AgenX_Nanohelix", "models", "nanohelix_scaler_X.pkl"
)
NANOHELIX_SCALER_Y_PATH = os.path.join(
    "AgenX_Nanohelix", "models", "nanohelix_scaler_y.pkl"
)
SUPERCON_MODEL_PATH = os.path.join(
    "AgenX_Supercon", "models", "best_supercon_model.pth"
)
SUPERCON_PROCESSOR_PATH = os.path.join(
    "AgenX_Supercon", "models", "supercon_processor.pkl"
)

# Global variables for loaded models
nanohelix_model = None
nanohelix_scaler_X = None
nanohelix_scaler_y = None
supercon_model = None
supercon_processor = None


def load_nanohelix_models():
    """Load Nanohelix models and scalers"""
    global nanohelix_model, nanohelix_scaler_X, nanohelix_scaler_y
    try:
        if all(
            os.path.exists(p)
            for p in [
                NANOHELIX_MODEL_PATH,
                NANOHELIX_SCALER_X_PATH,
                NANOHELIX_SCALER_Y_PATH,
            ]
        ):
            nanohelix_model = joblib.load(NANOHELIX_MODEL_PATH)
            nanohelix_scaler_X = joblib.load(NANOHELIX_SCALER_X_PATH)
            nanohelix_scaler_y = joblib.load(NANOHELIX_SCALER_Y_PATH)
            logger.info("Nanohelix models loaded successfully")
            return True
        else:
            logger.error("Nanohelix model files not found")
            return False
    except Exception as e:
        logger.error(f"Failed to load Nanohelix models: {e}")
        return False


def load_supercon_models():
    """Load Superconductor models"""
    global supercon_model, supercon_processor
    try:
        if os.path.exists(SUPERCON_PROCESSOR_PATH) and os.path.exists(
            SUPERCON_MODEL_PATH
        ):
            # Load processor data to get input size
            with open(SUPERCON_PROCESSOR_PATH, "rb") as f:
                processor_data = pickle.load(f)
                input_size = processor_data["n_features"]

            # Load model
            supercon_model = TcPredictor(input_size=input_size)
            supercon_model.load_state_dict(
                torch.load(SUPERCON_MODEL_PATH, map_location="cpu")
            )
            supercon_model.eval()

            # Initialize processor with correct working directory context
            supercon_processor = SuperconDataProcessor()
            # Pre-load the processor data to avoid path issues during inference
            supercon_processor.scaler = processor_data["scaler"]
            supercon_processor.elements = processor_data["elements"]
            supercon_processor.structure_types = processor_data["structure_types"]
            supercon_processor.feature_columns = processor_data["feature_columns"]
            supercon_processor._processor_data = processor_data  # Store for later use

            logger.info(
                f"Superconductor model loaded successfully with input size: {input_size}"
            )
            return True
        else:
            logger.error("Superconductor model files not found")
            return False
    except Exception as e:
        logger.error(f"Failed to load Superconductor models: {e}")
        return False


def compute_nanohelix_parameters(df):
    """Compute derived parameters for nanohelix prediction"""
    df_enriched = df.copy()

    pitch = df_enriched["pitch"]
    fiber_radius = df_enriched["fiber_radius"]
    n_turns = df_enriched["n_turns"]
    helix_radius = df_enriched["helix_radius"]

    turn_length = np.sqrt((2 * np.pi * helix_radius) ** 2 + pitch**2)

    df_enriched["total_length"] = turn_length * n_turns
    df_enriched["height"] = pitch * n_turns
    df_enriched["curl"] = helix_radius / (helix_radius**2 + (pitch / (2 * np.pi)) ** 2)
    df_enriched["angle"] = np.arctan2(pitch, 2 * np.pi * helix_radius)
    df_enriched["total_fiber_length"] = df_enriched["total_length"] * (
        1 + (2 * np.pi * fiber_radius) / turn_length
    )
    df_enriched["V"] = np.pi * fiber_radius**2 * df_enriched["total_fiber_length"]
    df_enriched["mass"] = df_enriched["V"]

    return df_enriched


@app.route("/prediction_chembl", methods=["POST"])
def predict_chembl():
    """
    ChEMBL prediction endpoint for SMILES strings.

    Expected JSON input:
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O"
    }
    """
    response = {"input": "", "output": "", "success": False, "error": ""}

    try:
        data = request.get_json()
        if not data:
            response["error"] = "No JSON data provided"
            return jsonify(response), 400

        smiles = data.get("smiles")
        logger.debug(f"Received SMILES: {smiles}")

        if not smiles:
            response["error"] = "SMILES string is missing in request body"
            return jsonify(response), 400

        # Make prediction using ChEMBL model
        predicted = predict_smiles(smiles, model_path=CHEMBL_MODEL_PATH)
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
        logger.error(f"Error during ChEMBL prediction: {str(e)}")
        response["error"] = f"Internal server error: {str(e)}"
        return jsonify(response), 500


@app.route("/prediction_nanohelix", methods=["POST"])
def predict_nanohelix():
    """
    Nanohelix g-factor prediction endpoint.

    Expected JSON input:
    {
        "pitch": 100,
        "fiber_radius": 30,
        "n_turns": 5,
        "helix_radius": 50
    }
    """
    response = {"input": "", "output": "", "success": False, "error": ""}

    try:
        if nanohelix_model is None:
            response["error"] = "Nanohelix model not loaded"
            return jsonify(response), 500

        data = request.get_json()
        logger.debug(f"Received nanohelix data: {data}")

        if not data:
            response["error"] = "No JSON data provided"
            return jsonify(response), 400

        # Check required parameters
        required_params = ["pitch", "fiber_radius", "n_turns", "helix_radius"]
        missing_params = [param for param in required_params if param not in data]

        if missing_params:
            response["error"] = (
                f"Missing required parameters: {', '.join(missing_params)}"
            )
            return jsonify(response), 400

        # Create DataFrame with input parameters
        input_df = pd.DataFrame(
            {
                "pitch": [data["pitch"]],
                "fiber_radius": [data["fiber_radius"]],
                "n_turns": [data["n_turns"]],
                "helix_radius": [data["helix_radius"]],
            }
        )

        # Compute derived parameters
        data_enriched = compute_nanohelix_parameters(input_df)

        # Get expected feature names from scaler
        expected_feature_names = nanohelix_scaler_X.feature_names_in_

        # Add missing features with default values
        for feature in expected_feature_names:
            if feature not in data_enriched.columns:
                data_enriched[feature] = 0

        # Ensure columns are in correct order
        X = data_enriched[expected_feature_names]

        # Scale features and make prediction
        X_scaled = nanohelix_scaler_X.transform(X)
        y_pred_scaled = nanohelix_model.predict(X_scaled)
        g_factor = nanohelix_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[
            0
        ][0]

        response["input"] = data
        response["output"] = float(g_factor)
        response["success"] = True
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during Nanohelix prediction: {str(e)}")
        response["error"] = f"Internal server error: {str(e)}"
        return jsonify(response), 500


def process_supercon_input(input_data, processor_data):
    """Custom input processing for superconductor that doesn't rely on file loading"""
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Check required columns
    if "element" not in input_data.columns:
        raise ValueError("Missing required field: 'element' (chemical formula)")

    # Get data from pre-loaded processor
    scaler = processor_data["scaler"]
    elements = processor_data["elements"]
    used_elements = processor_data["used_elements"]
    structure_types = processor_data["structure_types"]
    expected_features = processor_data["n_features"]

    # Create feature matrix matching the training data format
    num_samples = len(input_data)
    X = np.zeros((num_samples, expected_features))

    for i, formula in enumerate(input_data["element"]):
        # Process chemical formula
        element_counts = parse_formula(formula)
        total_count = sum(element_counts.values())

        if total_count > 0:  # Avoid division by zero
            # Fill element features
            for j, element in enumerate(used_elements):
                if element in element_counts:
                    X[i, j] = element_counts[element] / total_count

        # Fill structure type features if available
        if "str3" in input_data.columns and structure_types:
            struct = input_data.iloc[i]["str3"]
            if not pd.isna(struct):
                struct_str = str(struct)
                if struct_str in structure_types:
                    idx = structure_types.index(struct_str)
                    X[i, len(used_elements) + idx] = 1

    # Scale features
    scaled_data = scaler.transform(X)
    return scaled_data


def parse_formula(formula):
    """Parse chemical formula into elemental composition ratios"""
    import re

    element_counts = {}

    # Handle NaN values
    if pd.isna(formula):
        return element_counts

    # Convert to string just in case
    formula = str(formula)

    # Remove any non-standard characters
    formula = formula.replace("-", "")

    # Regular expression to match elements and their counts
    pattern = r"([A-Z][a-z]*)(\d*\.?\d*)"
    matches = re.findall(pattern, formula)

    # Sum up elemental counts
    for element, count in matches:
        if element in element_counts:
            element_counts[element] += float(count) if count else 1.0
        else:
            element_counts[element] = float(count) if count else 1.0

    return element_counts


@app.route("/prediction_supercon", methods=["POST"])
def predict_supercon():
    """
    Superconductor critical temperature prediction endpoint.

    Expected JSON input:
    {
        "element": "Ba0.2La1.8Cu1O4-Y",
        "str3": "T"
    }
    """
    response = {"input": "", "output": "", "success": False, "error": ""}

    try:
        if supercon_model is None or supercon_processor is None:
            response["error"] = "Superconductor model not loaded"
            return jsonify(response), 500

        data = request.get_json()
        logger.debug(f"Received superconductor data: {data}")

        if not data:
            response["error"] = "No JSON data provided"
            return jsonify(response), 400

        if "element" not in data:
            response["error"] = "Missing required field: 'element' (chemical formula)"
            return jsonify(response), 400

        # Process input data using custom function
        input_df = pd.DataFrame([data])
        processed_input = process_supercon_input(
            input_df, supercon_processor._processor_data
        )
        input_tensor = torch.FloatTensor(processed_input)

        # Make prediction
        with torch.no_grad():
            prediction = supercon_model(input_tensor)
            predicted_tc = float(prediction[0][0])

        logger.debug(f"Predicted Tc: {predicted_tc}")

        response["input"] = data
        response["output"] = predicted_tc
        response["success"] = True
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during Superconductor prediction: {str(e)}")
        response["error"] = f"Internal server error: {str(e)}"
        return jsonify(response), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    models_status = {
        "chembl": os.path.exists(CHEMBL_MODEL_PATH),
        "nanohelix": nanohelix_model is not None,
        "supercon": supercon_model is not None,
    }

    return jsonify(
        {
            "status": "healthy",
            "models": models_status,
            "endpoints": [
                "/prediction_chembl",
                "/prediction_nanohelix",
                "/prediction_supercon",
            ],
        }
    )


def initialize_models():
    """Initialize all models at startup"""
    logger.info("Initializing models...")

    # Load Nanohelix models
    if not load_nanohelix_models():
        logger.warning("Nanohelix models could not be loaded")

    # Load Superconductor models
    if not load_supercon_models():
        logger.warning("Superconductor models could not be loaded")

    # Check ChEMBL model
    if os.path.exists(CHEMBL_MODEL_PATH):
        logger.info("ChEMBL model path found")
    else:
        logger.warning("ChEMBL model not found")


if __name__ == "__main__":
    # Change to envs directory to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Initialize models
    initialize_models()

    logger.info(f"Starting unified prediction server on {HOST}:{PORT}")
    logger.info("Available endpoints:")
    logger.info("  POST /prediction_chembl - ChEMBL SMILES prediction")
    logger.info("  POST /prediction_nanohelix - Nanohelix g-factor prediction")
    logger.info("  POST /prediction_supercon - Superconductor Tc prediction")
    logger.info("  GET /health - Health check")

    app.run(debug=True, host=HOST, port=PORT)
