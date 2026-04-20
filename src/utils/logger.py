import os
import time
import mlflow
from functools import wraps

# Setup MLflow Tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "query_understanding_pipeline"))

def trace_execution(step_name: str):
    """
    Decorator to trace latency, inputs, and outputs of pipeline steps.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            with mlflow.start_run(nested=True, run_name=step_name):
                # Log inputs
                mlflow.log_dict({"args": args, "kwargs": kwargs}, f"{step_name}_inputs.json")
                
                try:
                    result = func(*args, **kwargs)
                    latency = time.time() - start_time
                    
                    # Log performance and outputs
                    mlflow.log_metric(f"{step_name}_latency_sec", latency)
                    mlflow.log_dict({"result": result}, f"{step_name}_outputs.json")
                    return result
                except Exception as e:
                    mlflow.log_param(f"{step_name}_error", str(e))
                    raise e
        return wrapper
    return decorator