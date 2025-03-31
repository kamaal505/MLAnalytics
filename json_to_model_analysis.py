import os
import json
import pandas as pd
from collections import defaultdict

def load_json(file_path):
    """Load JSON data from a given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_data(data):
    """Identify faulty conversation IDs and perform probability analysis."""
    faulty_conversation_ids = set()
    model_stats = {'A': {'success': 0, 'failure': 0}, 'B': {'success': 0, 'failure': 0}}
    prompt_type_failures = defaultdict(lambda: {'A': 0, 'B': 0})
    prompt_type_counts = defaultdict(int)
    error_type_counts = {'A': defaultdict(int), 'B': defaultdict(int)}
    
    for conversation in data:
        conversation_id = conversation.get("conversationId", "Unknown")  # Fixed key name
        prompt_evaluations = conversation.get("promptEvaluations", {})
        prompt_type = prompt_evaluations.get("prompt type", "Unknown")
        prompt_type_counts[prompt_type] += 1
        evaluations = conversation.get("modelEvaluations", [])
        
        if len(evaluations) < 2:
            continue  # Ensure two models exist
        
        for idx, model_key in enumerate(['A', 'B']):
            eval_entry = evaluations[idx]
            model_break = eval_entry.get("model break", "False") == "True"
            error_type = eval_entry.get("error type", "Unknown")
            
            if model_break:
                if error_type == "n/a":
                    faulty_conversation_ids.add(conversation_id)
                    continue  # Skip faulty conversation entries
                model_stats[model_key]['failure'] += 1
                prompt_type_failures[prompt_type][model_key] += 1
                error_type_counts[model_key][error_type] += 1
            else:
                model_stats[model_key]['success'] += 1
    
    return model_stats, prompt_type_failures, error_type_counts, faulty_conversation_ids, prompt_type_counts

def compute_probabilities(model_stats, prompt_type_failures, error_type_counts):
    """Compute probability distributions."""
    total_A = sum(model_stats['A'].values())
    total_B = sum(model_stats['B'].values())
    
    prob_model = {
        'A': {k: round((v / total_A) * 100, 2) for k, v in model_stats['A'].items() if total_A > 0},
        'B': {k: round((v / total_B) * 100, 2) for k, v in model_stats['B'].items() if total_B > 0}
    }
    
    prob_prompt_type = {
        prompt: {m: round((v / sum(pt.values())) * 100, 2) for m, v in pt.items() if sum(pt.values()) > 0} 
        for prompt, pt in prompt_type_failures.items()
    }
    
    prob_error_type = {
        'A': {k: round((v / sum(error_type_counts['A'].values())) * 100, 2) for k, v in error_type_counts['A'].items() if sum(error_type_counts['A'].values()) > 0},
        'B': {k: round((v / sum(error_type_counts['B'].values())) * 100, 2) for k, v in error_type_counts['B'].items() if sum(error_type_counts['B'].values()) > 0}
    }
    
    return {
        "prob_model": prob_model,
        "prob_prompt_type": prob_prompt_type,
        "prob_error_type": prob_error_type
    }

def save_results(results, prompt_type_counts, prompt_type_failures, output_dir):
    """Save results as CSV and JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in results.items():
        json_path = os.path.join(output_dir, f"{name}.json")
        csv_path = os.path.join(output_dir, f"{name}.csv")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        df = pd.DataFrame.from_dict(data, orient='index')
        df.to_csv(csv_path)
    
    prob_prompt_type_with_counts = results["prob_prompt_type"].copy()
    for prompt, counts in prompt_type_counts.items():
        if prompt in prob_prompt_type_with_counts:
            prob_prompt_type_with_counts[prompt]["total_count"] = counts
            prob_prompt_type_with_counts[prompt]["failure_count"] = prompt_type_failures[prompt]["A"] + prompt_type_failures[prompt]["B"]
    
    json_path = os.path.join(output_dir, "prob_prompt_type_with_counts.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prob_prompt_type_with_counts, f, indent=4)

def main():
    file_path = input("Enter the path to the JSON file: ")
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    
    output_dir = os.path.join(os.path.dirname(file_path), "falcon_analysis")
    data = load_json(file_path)
    model_stats, prompt_type_failures, error_type_counts, faulty_conversation_ids, prompt_type_counts = analyze_data(data)
    results = compute_probabilities(model_stats, prompt_type_failures, error_type_counts)
    
    # Save faulty conversation IDs separately
    faulty_ids_path = os.path.join(output_dir, "faulty_conversation_ids.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(faulty_ids_path, 'w', encoding='utf-8') as f:
        json.dump(list(faulty_conversation_ids), f, indent=4)
    
    print(f"Faulty conversation IDs saved in {faulty_ids_path}")
    save_results(results, prompt_type_counts, prompt_type_failures, output_dir)
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
