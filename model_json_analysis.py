import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_probabilities(data, variable, group_by):
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df[df[group_by].notna() & (df[group_by] != "")]  # Remove empty group_by entries
    probability_df = df.groupby(group_by)[variable].value_counts(normalize=True).unstack().fillna(0)
    probability_df.rename(columns={'': 'Model Success'}, inplace=True)
    probability_df = (probability_df * 100).round(2)  # Convert to percentages and round to 2 decimal places
    return probability_df

def save_csv_json(probability_df, output_dir, filename):
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    json_path = os.path.join(output_dir, f"{filename}.json")
    probability_df.to_csv(csv_path, index=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(probability_df.to_dict(orient='index'), f, indent=4)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")

def sanitize_filename(name):
    name = name.strip().replace("\n", "").replace(" ", "_")
    return re.sub(r'[\/:*?"<>|]', '_', name)

def plot_bar_chart(probability_df, output_dir, variable, group_by):
    plt.figure(figsize=(16, 9))  # Large resolution for full-screen clarity
    probability_df.plot(kind='bar', stacked=True, colormap='tab10', edgecolor='black')
    plt.xlabel(group_by, fontsize=14)
    plt.ylabel("Probability (%)", fontsize=14)
    plt.title(f"{variable} Probability Distribution by {group_by}", fontsize=16)
    plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{variable}_by_{group_by}.png")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart: {output_file}")

def plot_pie_chart(probability_df, output_dir, group_by):
    for category in probability_df.index:
        sanitized_category = sanitize_filename(category)
        plt.figure(figsize=(10, 10))  # Large size for clarity
        probability_df.loc[category][probability_df.loc[category] > 0].plot(kind='pie', 
            autopct='%1.1f%%', startangle=140, cmap='tab10')
        plt.ylabel('')
        plt.title(f"Error Type Distribution for {category}", fontsize=14)
        output_file = os.path.join(output_dir, f"error_type_pie_{sanitized_category}.png")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"Saved pie chart: {output_file}")

def main():
    file_path = input("Enter the JSON file path: ").strip()
    if not os.path.exists(file_path):
        print("Invalid file path.")
        return
    
    data = load_json(file_path)
    output_dir = os.path.dirname(file_path)
    
    # Generate required probability distributions
    required_categories = [
        ("model_break_scenario", "complexity"),
        ("model_break_scenario", "prompt_type"),
        ("error_type", "prompt_type")
    ]
    
    for variable, category in required_categories:
        probability_df = compute_probabilities(data, variable, category)
        save_csv_json(probability_df, output_dir, f"{variable}_by_{category}")
        plot_bar_chart(probability_df, output_dir, variable, category)
    
    # Generate pie charts only for error_type_by_prompt_type
    error_prob_df = compute_probabilities(data, "error_type", "prompt_type")
    plot_pie_chart(error_prob_df, output_dir, "prompt_type")

if __name__ == "__main__":
    main()
