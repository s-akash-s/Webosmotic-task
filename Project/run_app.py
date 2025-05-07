# run_app.py
import subprocess
import time
import os
import sys

def run_fastapi_server():
    """Run the FastAPI server"""
    print("Starting FastAPI server...")
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"])
    # Wait for server to start
    time.sleep(10)  # Give it 10 seconds to fully initialize
    print("FastAPI server is running on http://localhost:8000")
    return process

def run_streamlit_app():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"])
    print("Streamlit app is running")
    return process

if __name__ == "__main__":
    # Create directory for UI if it doesn't exist
    os.makedirs("app/ui", exist_ok=True)
    
    # Run servers
    fastapi_process = run_fastapi_server()
    streamlit_process = run_streamlit_app()
    
    print("Both servers are running. Press Ctrl+C to stop.")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping servers...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)