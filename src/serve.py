"""
Inference API Server - Entry Point

Run the server from the repository root:

    # Default
    python -m src.serve

    # Custom host and port
    python -m src.serve --host 0.0.0.0 --port 9000

    # Preload specific models
    python -m src.serve --preload transvert_femur swinx2s_hip
"""

import sys
import os

# Ensure the repository root is first on sys.path so `python src/serve.py` also
# works when launched with the repo root as the working directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Start the inference server
from src.inference.run_server import main

if __name__ == "__main__":
    main()
