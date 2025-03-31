import json
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_json(file_path):
    """
    Load and return the JSON data from the given file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_label(label):
    """
    Normalize a label by converting it to lowercase, trimming spaces,
    and replacing spaces between words with underscores.
    """
    if not isinstance(label, str):
        return label
    norm = label.lower().strip()
    norm = norm.replace(" ", "_")
    return norm


def calculate_failure_percentage(data):
    """
    For every model in the data, compute the failure percentage as the proportion of evaluations
    that returned 'yes' (indicating failure). Returns a dict keyed by model ID.
    """
    evaluations_by_model = {}
    
    for entry in data:
        model_configs = entry.get("modelConfigs", [])
        model_evaluations = entry.get("modelEvaluations", [])
        
        for model in model_configs:
            model_id = model.get("modelId")
            if model_id is None:
                continue
            if model_id not in evaluations_by_model:
                evaluations_by_model[model_id] = []
            for eval_entry in model_evaluations:
                if not isinstance(eval_entry, dict):
                    continue
                # Use the modelId to match evaluations to model config.
                eval_model_id = eval_entry.get("modelId")
                if eval_model_id == model_id:
                    failure_value = eval_entry.get("model failure", "No")
                    evaluations_by_model[model_id].append(failure_value.strip().lower())
    
    model_failures = {}
    for model_id, failures in evaluations_by_model.items():
        if failures:
            percent = (failures.count("yes") / len(failures)) * 100
            model_failures[model_id] = round(percent, 2)
        else:
            model_failures[model_id] = 0
    return model_failures


def save_failure_percentages_to_json(failure_percentages, output_file):
    """
    Write the overall model failure percentages to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(failure_percentages, f, indent=4)
    print(f"Failure percentages JSON saved at: {output_file}")


def create_failure_distribution(data):
    """
    Group the evaluation data by model and subject.
    Normalize subject labels so that minor variations are combined.
    Returns a dictionary like: {model_id: {subject: {'yes': count, 'no': count}}}.
    """
    distribution = {}
    
    for entry in data:
        prompt_evaluations = entry.get("promptEvaluations", [])
        model_evaluations = entry.get("modelEvaluations", [])
        
        # Ensure both evaluations are lists.
        if not isinstance(prompt_evaluations, list):
            prompt_evaluations = [prompt_evaluations]
        if not isinstance(model_evaluations, list):
            model_evaluations = [model_evaluations]
        
        for eval_entry in model_evaluations:
            if not isinstance(eval_entry, dict):
                continue
            failure_value = eval_entry.get("model failure", "No").strip().lower()
            model_id = eval_entry.get("modelId", "Unknown")
            
            for prompt in prompt_evaluations:
                if not isinstance(prompt, dict):
                    continue
                # Normalize subject names.
                subject = normalize_label(prompt.get("subject", "Unknown"))
                if model_id not in distribution:
                    distribution[model_id] = {}
                if subject not in distribution[model_id]:
                    distribution[model_id][subject] = {"yes": 0, "no": 0}
                distribution[model_id][subject][failure_value] += 1
    return distribution


def create_failure_distribution_by_complexity(data):
    """
    Group evaluation data by model and complexity.
    Normalize complexity labels.
    Returns: {model_id: {complexity: {'yes': count, 'no': count}}}.
    """
    distribution = {}
    
    for entry in data:
        prompt_evaluations = entry.get("promptEvaluations", [])
        model_evaluations = entry.get("modelEvaluations", [])
        
        if not isinstance(prompt_evaluations, list):
            prompt_evaluations = [prompt_evaluations]
        if not isinstance(model_evaluations, list):
            model_evaluations = [model_evaluations]
        
        for eval_entry in model_evaluations:
            if not isinstance(eval_entry, dict):
                continue
            failure_value = eval_entry.get("model failure", "No").strip().lower()
            model_id = eval_entry.get("modelId", "Unknown")
            
            for prompt in prompt_evaluations:
                if not isinstance(prompt, dict):
                    continue
                complexity = normalize_label(prompt.get("complexity", prompt.get("promptEvaluations.complexity", "Unknown")))
                if model_id not in distribution:
                    distribution[model_id] = {}
                if complexity not in distribution[model_id]:
                    distribution[model_id][complexity] = {"yes": 0, "no": 0}
                distribution[model_id][complexity][failure_value] += 1
                
    return distribution


def save_failure_distribution_to_json(distribution, output_file):
    """
    Save the distribution (grouped by model and subject) into a JSON file.
    The JSON structure has each model with two keys: 'model failure' (yes counts)
    and 'model success' (no counts) per subject.
    """
    nested_json = {}
    for model_id, subjects in distribution.items():
        nested_json[model_id] = {"model failure": {}, "model success": {}}
        for subject, counts in subjects.items():
            nested_json[model_id]["model failure"][subject] = counts.get("yes", 0)
            nested_json[model_id]["model success"][subject] = counts.get("no", 0)
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(nested_json, f, indent=4)
    print(f"Nested JSON saved at: {output_file}")


def save_failure_distribution_to_csv_subject(distribution, output_csv, output_json):
    """
    Create a CSV and a JSON where each row represents a subject.
    The CSV has:
      - 'Count': total instances (divided by number of models)
      - 'Probability of Model Failure': aggregated probability (failures/total) as a percentage
      - Subsequent columns (one per model) with each model’s failure probability as a percentage.
    The JSON is structured with each subject as a key, and its value is a dictionary of the remaining data.
    """
    models = list(distribution.keys())
    subjects = set()
    for model_id, subject_counts in distribution.items():
        subjects.update(subject_counts.keys())
    
    # Data containers for CSV/JSON
    data_rows = {}
    
    for subject in subjects:
        agg_yes = 0
        agg_total = 0
        row = {}
        for model_id in models:
            counts = distribution.get(model_id, {}).get(subject, {"yes": 0, "no": 0})
            total = counts.get("yes", 0) + counts.get("no", 0)
            agg_yes += counts.get("yes", 0)
            agg_total += total
            # Express as percentage.
            prob = (counts.get("yes", 0) / total * 100) if total > 0 else 0
            row[model_id] = round(prob, 2)
        adjusted_count = agg_total / len(models) if models else 0
        agg_prob = (agg_yes / agg_total * 100) if agg_total > 0 else 0
        row["Count"] = int(adjusted_count)
        row["Probability of Model Failure"] = round(agg_prob, 2)
        data_rows[subject] = row

    # Create DataFrame with subjects as index in desired order.
    df = pd.DataFrame.from_dict(data_rows, orient="index")
    model_columns = sorted(models)
    df = df[["Count", "Probability of Model Failure"] + model_columns]
    df.index.name = "Subject"
    
    df.to_csv(output_csv)
    print(f"CSV file (by subject) saved at: {output_csv}")
    
    # Create JSON with subject as key.
    result_json = df.to_dict(orient="index")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)
    print(f"JSON file (by subject) saved at: {output_json}")


def save_failure_distribution_to_csv_complexity(distribution, output_csv, output_json):
    """
    Create a CSV and a JSON where each row represents a complexity level.
    The CSV has:
      - 'Count': total instances (divided by number of models)
      - For each model, the failure probability (yes/(yes+no)) as a percentage.
    The JSON is structured with each complexity level as a key, and its value is a dictionary containing count and each model's probability.
    """
    models = list(distribution.keys())
    complexities = set()
    for model_id, comp_counts in distribution.items():
        complexities.update(comp_counts.keys())
    
    data_rows = {}
    for comp in complexities:
        row = {}
        total_count = 0
        for model_id in models:
            counts = distribution.get(model_id, {}).get(comp, {"yes": 0, "no": 0})
            total = counts.get("yes", 0) + counts.get("no", 0)
            total_count += total
            prob = (counts.get("yes", 0) / total * 100) if total > 0 else 0
            row[model_id] = round(prob, 2)
        adjusted_count = total_count / len(models) if models else 0
        row["Count"] = int(adjusted_count)
        data_rows[comp] = row

    df = pd.DataFrame.from_dict(data_rows, orient="index")
    model_columns = sorted(models)
    df = df[["Count"] + model_columns]
    df.index.name = "Complexity"
    
    df.to_csv(output_csv)
    print(f"CSV file (by complexity) saved at: {output_csv}")
    
    # Create JSON with complexity as key.
    result_json = df.to_dict(orient="index")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)
    print(f"JSON file (by complexity) saved at: {output_json}")


def save_conditional_failure_distribution(distribution, output_csv, output_json, output_chart):
    """
    For each model, calculate the conditional probability of failure for each complexity level
    given that the model failed. Express probabilities as percentages.
    Create a CSV, JSON, and a stacked bar chart for this data.
    """
    models = list(distribution.keys())
    complexities = set()
    for model_id, comp_counts in distribution.items():
        complexities.update(comp_counts.keys())
    complexities = sorted(list(complexities))
    
    conditional_data = {}
    
    for model_id in models:
        conditional_data[model_id] = {}
        total_failures = 0
        for comp in complexities:
            counts = distribution.get(model_id, {}).get(comp, {"yes": 0})
            total_failures += counts.get("yes", 0)
        for comp in complexities:
            counts = distribution.get(model_id, {}).get(comp, {"yes": 0})
            cond_prob = (counts.get("yes", 0) / total_failures * 100) if total_failures > 0 else 0
            conditional_data[model_id][comp] = round(cond_prob, 2)
    
    df = pd.DataFrame.from_dict(conditional_data, orient="index")
    df.index.name = "ModelID"
    
    df.to_csv(output_csv)
    print(f"CSV file (conditional failure by complexity) saved at: {output_csv}")
    
    result_json = df.to_dict(orient="index")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)
    print(f"JSON file (conditional failure by complexity) saved at: {output_json}")
    
    ax = df.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_xlabel("ModelID")
    ax.set_ylabel("Conditional Probability of Failure (given failure) [%]")
    ax.set_title("Conditional Failure Probability by Complexity for each Model")
    plt.legend(title="Complexity", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_chart)
    plt.close()
    print(f"Bar chart saved at: {output_chart}")


def main():
    # Ask for input file path
    file_path = input("Enter the full path of the input JSON file: ").strip()
    
    # Ensure the file exists
    if not os.path.isfile(file_path):
        print("Error: The specified file does not exist.")
        return
    
    # Define the output directory (same location as input file)
    input_directory = Path(file_path).parent
    output_directory = input_directory / "benchmarking_data"

    # Create the directory if it doesn't exist
    output_directory.mkdir(exist_ok=True)

    # Define output file paths within the new directory
    output_failure_json = output_directory / "failure_percentages.json"
    output_nested_json = output_directory / "model_failure_distribution.json"
    output_csv_subject = output_directory / "model_failure_distribution_by_subject.csv"
    output_json_subject = output_directory / "model_failure_distribution_by_subject.json"
    output_csv_complexity = output_directory / "model_failure_distribution_by_complexity.csv"
    output_json_complexity = output_directory / "model_failure_distribution_by_complexity.json"
    output_csv_conditional = output_directory / "conditional_failure_distribution.csv"
    output_json_conditional = output_directory / "conditional_failure_distribution.json"
    output_chart = output_directory / "conditional_failure_distribution_chart.png"

    # Load the input JSON
    data = load_json(file_path)

    # Calculate and save overall failure percentages
    failure_percentages = calculate_failure_percentage(data)
    print("Failure Percentages by Model ID:")
    print(json.dumps(failure_percentages, indent=4))
    save_failure_percentages_to_json(failure_percentages, output_failure_json)

    # Create and save distribution data (by subject)
    distribution_subject = create_failure_distribution(data)
    save_failure_distribution_to_json(distribution_subject, output_nested_json)
    save_failure_distribution_to_csv_subject(distribution_subject, output_csv_subject, output_json_subject)

    # Create and save distribution data (by complexity)
    distribution_complexity = create_failure_distribution_by_complexity(data)
    save_failure_distribution_to_csv_complexity(distribution_complexity, output_csv_complexity, output_json_complexity)

    # Create and save conditional failure distribution (given failure) by complexity
    save_conditional_failure_distribution(distribution_complexity, output_csv_conditional, output_json_conditional, output_chart)

    print(f"\nAll output files are saved in: {output_directory}")


if __name__ == "__main__":
    main()
