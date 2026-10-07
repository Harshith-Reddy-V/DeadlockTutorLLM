"""Development runner script for DeadlockTutorLLM."""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def run_backend():
    print("Starting DeadlockTutorLLM FastAPI Backend on port 8000...")
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"], cwd=BASE_DIR)


def run_frontend():
    print("Starting DeadlockTutorLLM Streamlit Frontend on port 8501...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/app.py"], cwd=BASE_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "frontend":
        run_frontend()
    else:
        run_backend()
