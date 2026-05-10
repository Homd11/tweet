"""
Run the Streamlit dashboard.
Usage: python run_dashboard.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "src/dashboard/app.py",
    "--server.port", "8501",
    "--server.address", "localhost",
])