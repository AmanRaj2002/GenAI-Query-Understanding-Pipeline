import subprocess
import os
import time
import signal
import sys

# --- CONFIGURATION ---
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
BASE_DIR = os.getcwd()
MODEL_PATH = f"{BASE_DIR}/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Define Environment Variables
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = f"{CONDA_PREFIX}/lib:{ENV.get('LD_LIBRARY_PATH', '')}"
ENV["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5050"
ENV["CUDA_VISIBLE_DEVICES"] = "0,1"
ENV["PYTHONPATH"] = BASE_DIR

processes = []

def start_process(command, name, log_file):
    print(f"Starting {name}...")
    log = open(log_file, "w")
    proc = subprocess.Popen(
        command,
        shell=True,
        env=ENV,
        stdout=log,
        stderr=log,
        preexec_fn=os.setsid # Ensures child processes die when this script dies
    )
    processes.append((name, proc))
    return proc

def cleanup(sig, frame):
    print("\nShutting down all services...")
    for name, proc in processes:
        print(f"Killing {name}...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    sys.exit(0)

# Register cleanup for CTRL+C
signal.signal(signal.SIGINT, cleanup)

# --- EXECUTION ---
print("🧹 Cleaning up old processes...")
subprocess.run("pkill -9 -f 'mlflow|uvicorn|redis-stack|streamlit'", shell=True)
time.sleep(2)

# 1. Start Redis
start_process("./redis-stack-server-7.2.0-v9/bin/redis-stack-server", "Redis", "redis.log")

# 2. Start MLflow (Port 5050 to avoid SIGTERM issues)
start_process("mlflow server --host 127.0.0.1 --port 5050 --workers 1", "MLflow", "mlflow.log")

# 3. Start API (FastAPI)
start_process("python -u -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000", "FastAPI", "api.log")

# 4. Start UI (Streamlit)
start_process("streamlit run src/app/ui.py --server.port 8501", "Streamlit", "streamlit.log")

print("\n All systems are booting!")
print("API: http://localhost:8000")
print("UI:  http://localhost:8501")
print("Tracking: http://localhost:5050")
print("\n Monitoring API Logs (Press CTRL+C to stop all)...")

# Keep the script alive and tail the API log
try:
    with subprocess.Popen(["tail", "-f", "api.log"]) as tail:
        tail.wait()
except KeyboardInterrupt:
    cleanup(None, None)