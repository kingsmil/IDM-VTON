import runpod
import uvicorn
import subprocess
import logging
import time
import requests
from threading import Thread

# Local imports (ensure Python path includes the app directory)
from tryon_logic import initialize_models # Import the initializer

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global State ---
server_process = None
is_server_ready = False

def check_server_ready(url, timeout=300):
    """Polls the health check endpoint until the server is ready."""
    global is_server_ready
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("FastAPI server is ready.")
                is_server_ready = True
                return
        except requests.exceptions.RequestException as e:
            logger.debug(f"Server not ready yet ({e}), waiting...")
        time.sleep(2)
    logger.error(f"Server did not become ready within {timeout} seconds.")
    is_server_ready = False # Explicitly set to false on timeout


def start_server():
    """Starts the Uvicorn server as a subprocess and waits for it."""
    global server_process
    # Pre-load models BEFORE starting the server
    try:
        logger.info("Handler: Initializing models...")
        initialize_models() # Call the function that loads everything
        logger.info("Handler: Model initialization complete.")
    except Exception as e:
        logger.critical(f"Handler: CRITICAL - Failed to initialize models: {e}", exc_info=True)
        # Optionally raise an error to prevent worker start, or log and continue
        return # Prevent server start if models failed

    # Start Uvicorn server
    logger.info("Handler: Starting Uvicorn server...")
    server_process = subprocess.Popen([
        "uvicorn", "app_api:app",
        "--host", "0.0.0.0",
        "--port", "8000", # Standard port
        "--workers", "1" # Important: RunPod manages scaling, use 1 worker per container
    ])

    # Start a thread to check readiness
    health_check_url = "http://127.0.0.1:8000/health"
    readiness_thread = Thread(target=check_server_ready, args=(health_check_url,), daemon=True)
    readiness_thread.start()


# --- RunPod Handler Function ---
def handler(job):
    """
    This function is called by RunPod for each job.
    It proxies the request to the running FastAPI application.
    """
    global is_server_ready, server_process

    if server_process is None or server_process.poll() is not None:
       logger.error("Server process is not running. Attempting restart?")
       # Handle restart logic if necessary, or fail the job
       return {"error": "Server process not running"}

    if not is_server_ready:
        logger.warning("Server not ready, job might fail or timeout.")
        # Optional: Wait a bit longer here, or return error immediately
        time.sleep(5) # Give it a bit more time
        if not is_server_ready:
           return {"error": "Server failed to become ready in time."}


    job_input = job['input']
    api_endpoint = "http://127.0.0.1:8000/tryon" # Endpoint defined in app_api.py

    try:
        logger.info(f"Proxying job {job['id']} to {api_endpoint}")
        response = requests.post(api_endpoint, json=job_input, timeout=600) # Adjust timeout as needed
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Job {job['id']} processed successfully by API.")
        return response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Job {job['id']} timed out waiting for API response.")
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Job {job['id']} failed: Error connecting to API or API error: {e}")
        # Try to get error detail from response if available
        error_detail = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except ValueError: # Not JSON
                 error_detail = e.response.text[:200] # First 200 chars
        return {"error": f"API request failed: {error_detail}"}
    except Exception as e:
        logger.error(f"Job {job['id']} failed: Unexpected error in handler: {e}", exc_info=True)
        return {"error": f"Unexpected handler error: {str(e)}"}


# --- Start Server on Worker Init ---
start_server()

# --- RunPod Entry Point ---
# This starts the RunPod worker loop
runpod.serverless.start({"handler": handler})