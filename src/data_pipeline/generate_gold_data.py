import os
import json
import pandas as pd
import mlflow
from src.utils.logger import trace_execution

# Setup MLflow Tracking for the data pipeline
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "query_understanding_pipeline"))

@trace_execution("golden_dataset_generation")
def generate_golden_dataset(raw_data_path: str, output_path: str):
    print(f"Reading raw data from: {raw_data_path}")

    # Mocking the reading of your robustsirr_test_dataset
    # In reality, you would load your unstructured search logs here
    raw_queries = [
        "epcot restaurants",
        "best vegan burritos near me open now",
        "cheap parking downtown",
        "watch the game tonight"
    ]

    # Mocking the Frontier LLM (e.g., GPT-4 / Claude) few-shot extraction process
    # This simulates the API calls that generate the high-quality labels
    print("Simulating Frontier LLM extraction for golden labels...")
    gold_records = []

    mock_llm_responses = [
        {"location": "epcot", "topic": "restaurants"},
        {"topic": "vegan burritos", "location": "near me", "time": "open now"},
        {"topic": "parking", "location": "downtown", "price": "cheap"},
        {"topic": "sports bar", "time": "tonight"}
    ]

    for query, parsed_intent in zip(raw_queries, mock_llm_responses):
        gold_records.append({
            "raw_query": query,
            "ground_truth_segmentation": json.dumps(parsed_intent)
        })

    df_gold = pd.DataFrame(gold_records)

    # Save the pristine dataset to the gold_dataset directory
    df_gold.to_csv(output_path, index=False)
    print(f"Golden dataset successfully saved to: {output_path}")

    # Log the dataset artifact to MLflow for strict data versioning
    mlflow.log_artifact(output_path, artifact_path="datasets")
    return len(df_gold)

if __name__ == "__main__":
    RAW_PATH = "data/raw/robustsirr_test_dataset"
    GOLD_PATH = "data/gold_dataset/training_data.csv"

    generate_golden_dataset(RAW_PATH, GOLD_PATH)
