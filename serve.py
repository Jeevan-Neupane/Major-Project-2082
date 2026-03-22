"""
Top-level entry point for the inference API server.
Run from any location:

    python serve.py
    python serve.py --host 0.0.0.0 --port 9000
    python serve.py --preload transvert_femur swinx2s_hip
"""
import sys
import os

# Always resolve the project root and put it first on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Delegate to the inference module's run_server main()
from inference.run_server import main

if __name__ == "__main__":
    main()
