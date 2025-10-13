#!/usr/bin/env python
"""
Startup script for the unified prediction server.
This script launches a single server containing all prediction models.
"""

import os
import sys


def main():
    """Launch the unified prediction server"""
    print("=" * 60)
    print("PiEvo Unified Prediction Server")
    print("=" * 60)
    print()
    print("Starting unified server with all prediction models...")
    print()
    print("Available endpoints:")
    print("  POST /prediction_chembl     - ChEMBL SMILES prediction")
    print("  POST /prediction_nanohelix  - Nanohelix g-factor prediction")
    print("  POST /prediction_supercon   - Superconductor Tc prediction")
    print("  GET  /health                - Health check")
    print()
    print("Server will be available at: http://127.0.0.1:12500")
    print("=" * 60)
    print()

    # Change to the envs directory
    envs_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(envs_dir)

    # Import and run the unified server
    try:
        from unified_server import app, initialize_models, HOST, PORT

        # Initialize all models
        initialize_models()

        # Start the server
        app.run(debug=False, host=HOST, port=PORT)

    except ImportError as e:
        print(f"Error importing unified server: {e}")
        print("Make sure you're running from the envs/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
