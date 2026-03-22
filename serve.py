"""
Inference API Server - Entry Point

Run the server from any location:

    # Default
    python serve.py

    # Custom host and port
    python serve.py --host 0.0.0.0 --port 9000

    # Preload specific models
    python serve.py --preload transvert_femur swinx2s_hip
"""

import sys
import os

# Ensure project root is first in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Start the inference server
from inference.run_server import main

if __name__ == "__main__":
    main()
