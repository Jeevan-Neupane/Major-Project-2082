"""
Entry point to start the inference API server.

Usage:
    python -m inference.run_server                    # default: localhost:8000
    python -m inference.run_server --host 0.0.0.0     # expose to network
    python -m inference.run_server --port 9000         # custom port
    python -m inference.run_server --preload swinx2s_femur transvert_hip
"""
import argparse
import sys
import os

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="X-Ray to 3D Inference API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload for development")
    parser.add_argument(
        "--preload", nargs="*", default=[],
        help="Model keys to preload at startup (e.g. swinx2s_femur transvert_hip)",
    )
    args = parser.parse_args()

    # Preload models before starting server
    if args.preload:
        from inference.predictor import Predictor
        from inference.api import _loaded_predictors
        for key in args.preload:
            print(f"[startup] preloading {key}...")
            _loaded_predictors[key] = Predictor(key)
            print(f"[startup] {key} ready")

    import uvicorn
    uvicorn.run(
        "inference.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
