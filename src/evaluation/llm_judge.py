import os
import requests
import mlflow
import pandas as pd

# 1. Force connection to your active MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("query_understanding_pipeline")

# 2. Golden Dataset based on your CSV
eval_data = pd.DataFrame({
    "inputs": [
        "epcot restaurants", 
        "best vegan burritos near me open now"
    ],
    "ground_truth": [
        "{'location': 'epcot', 'topic': 'restaurants'}", 
        "{'topic': 'vegan burritos', 'location': 'near me', 'time': 'open now'}"
    ]
})

# 3. Custom model function that hits your live API
def evaluate_api(df):
    predictions = []
    for query in df["inputs"]:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/parse_query",
                json={"query": query}
            )
            entities = response.json().get("extracted_entities", {})
            predictions.append(str(entities))
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
    return predictions

# 4. Create a custom, dependency-free exact match metric
def calculate_exact_match(eval_df, _builtin_metrics):
    # Compares the prediction column directly to the target (ground_truth) column
    score = (eval_df["prediction"] == eval_df["target"]).mean()
    return score

custom_exact_match = mlflow.metrics.make_metric(
    eval_fn=calculate_exact_match,
    greater_is_better=True,
    name="custom_exact_match"
)

# 5. Run the Evaluation
print("Starting evaluation run...")
with mlflow.start_run(run_name="api_end_to_end_eval"):
    results = mlflow.models.evaluate(
        model=evaluate_api,
        data=eval_data,
        targets="ground_truth",
        # We removed model_type="text" to stop the tiktoken/textstat crashes
        extra_metrics=[custom_exact_match] 
    )
    
    print("\n--- Evaluation Metrics ---")
    for metric_name, value in results.metrics.items():
        print(f"{metric_name}: {value:.4f}")