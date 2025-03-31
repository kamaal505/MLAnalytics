import json
import csv
import os
from collections import defaultdict

# Prompt the user for the input JSON file path.
input_path = input("Enter the full path of the input JSON file: ").strip()

# Use the folder of the input file to store output files.
output_folder = os.path.dirname(input_path)

# Load the JSON data from the input file using UTF-8 encoding.
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Helper function to rename model_break_scenario values.
def rename_model_break(val):
    # assuming the values are case-insensitive "yes" and "no"
    if isinstance(val, str) and val.lower() == "yes":
        return "model_failure"
    elif isinstance(val, str) and val.lower() == "no":
        return "model_success"
    else:
        return val

# -------------------------------
# 1. Probability distribution of prompt_type vs model_break_scenario
#    Ignore entries where prompt_type is an empty string.
prompt_type_break = defaultdict(lambda: {"count": 0, "model_failure": 0, "model_success": 0})

for entry in data.values():
    ptype = entry.get("prompt_type", "").strip()
    # Skip empty prompt types.
    if not ptype:
        continue
    mb = rename_model_break(entry.get("model_break_scenario"))
    if mb not in ["model_failure", "model_success"]:
        continue
    prompt_type_break[ptype]["count"] += 1
    prompt_type_break[ptype][mb] += 1

# Calculate percentages for each prompt_type.
prompt_type_break_prob = {}
for ptype, counts in prompt_type_break.items():
    total = counts["count"]
    prompt_type_break_prob[ptype] = {
        "count": total,
        "model_failure": round((counts["model_failure"] / total) * 100, 2),
        "model_success": round((counts["model_success"] / total) * 100, 2)
    }

# Write to JSON and CSV for distribution 1.
json_out1 = os.path.join(output_folder, 'model_break_scenario_by_prompt_type.json')
csv_out1 = os.path.join(output_folder, 'model_break_scenario_by_prompt_type.csv')
with open(json_out1, 'w', encoding='utf-8') as f:
    json.dump(prompt_type_break_prob, f, indent=4)
with open(csv_out1, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_type", "count", "model_failure (%)", "model_success (%)"])
    for ptype, stats in prompt_type_break_prob.items():
        writer.writerow([ptype, stats["count"], stats["model_failure"], stats["model_success"]])

# -------------------------------
# 2. Probability distribution of error_type vs prompt_type (ignore blank error_type)
error_type_vs_prompt = defaultdict(lambda: defaultdict(int))
prompt_error_counts = defaultdict(int)

for entry in data.values():
    ptype = entry.get("prompt_type")
    err_type = entry.get("error_type", "").strip()
    if not ptype or not err_type:
        continue  # ignore if prompt_type missing or error_type is blank
    error_type_vs_prompt[ptype][err_type] += 1
    prompt_error_counts[ptype] += 1

error_type_vs_prompt_prob = {}
for ptype, error_counts in error_type_vs_prompt.items():
    total = prompt_error_counts[ptype]
    error_type_vs_prompt_prob[ptype] = {
        err: round((count / total) * 100, 2) for err, count in error_counts.items()
    }

# Write to JSON and CSV for distribution 2.
json_out2 = os.path.join(output_folder, 'probability_error_type_vs_prompt_type.json')
csv_out2 = os.path.join(output_folder, 'probability_error_type_vs_prompt_type.csv')
with open(json_out2, 'w', encoding='utf-8') as f:
    json.dump(error_type_vs_prompt_prob, f, indent=4)
with open(csv_out2, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_type", "error_type", "probability (%)"])
    for ptype, error_probs in error_type_vs_prompt_prob.items():
        for err, prob in error_probs.items():
            writer.writerow([ptype, err, prob])

# -------------------------------
# 3. Probability distribution of complexity vs model_break_scenario.
#    Only consider entries where complexity is not blank.
complexity_break = defaultdict(lambda: {"total": 0, "model_failure": 0, "model_success": 0})

for entry in data.values():
    complexity = entry.get("complexity", "").strip()
    if not complexity:
        continue
    mb = rename_model_break(entry.get("model_break_scenario"))
    if mb not in ["model_failure", "model_success"]:
        continue
    complexity_break[complexity]["total"] += 1
    complexity_break[complexity][mb] += 1

complexity_break_prob = {}
for comp, counts in complexity_break.items():
    total = counts["total"]
    complexity_break_prob[comp] = {
        "model_failure": round((counts["model_failure"] / total) * 100, 2),
        "model_success": round((counts["model_success"] / total) * 100, 2)
    }

# Write to JSON and CSV for distribution 3.
json_out3 = os.path.join(output_folder, 'probability_complexity_vs_model_break.json')
csv_out3 = os.path.join(output_folder, 'probability_complexity_vs_model_break.csv')
with open(json_out3, 'w', encoding='utf-8') as f:
    json.dump(complexity_break_prob, f, indent=4)
with open(csv_out3, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["complexity", "model_failure (%)", "model_success (%)"])
    for comp, stats in complexity_break_prob.items():
        writer.writerow([comp, stats["model_failure"], stats["model_success"]])

# -------------------------------
# 4. Probability distribution of topic vs model_break_scenario.
#    Group by the renamed model break outcome and compute topic probabilities.
topic_break = {"model_failure": defaultdict(int), "model_success": defaultdict(int)}
topic_total = {"model_failure": 0, "model_success": 0}

for entry in data.values():
    topic = entry.get("topic", "").strip()
    mb = rename_model_break(entry.get("model_break_scenario"))
    if not topic or mb not in ["model_failure", "model_success"]:
        continue
    topic_break[mb][topic] += 1
    topic_total[mb] += 1

topic_break_prob = {"model_failure": {}, "model_success": {}}
for mb in ["model_failure", "model_success"]:
    total = topic_total[mb]
    for topic, count in topic_break[mb].items():
        topic_break_prob[mb][topic] = round((count / total) * 100, 2)

# Write to JSON and CSV for distribution 4.
json_out4 = os.path.join(output_folder, 'probability_topic_vs_model_break.json')
csv_out4 = os.path.join(output_folder, 'probability_topic_vs_model_break.csv')
with open(json_out4, 'w', encoding='utf-8') as f:
    json.dump(topic_break_prob, f, indent=4)
with open(csv_out4, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["model_break", "topic", "probability (%)"])
    for mb, topics in topic_break_prob.items():
        for topic, prob in topics.items():
            writer.writerow([mb, topic, prob])

# -------------------------------
# 5. Overall probability distribution for model_break outcomes.
#    Compute total counts and percentages for model_failure and model_success.
overall_counts = {"model_failure": 0, "model_success": 0, "total_count": 0}

for entry in data.values():
    mb = rename_model_break(entry.get("model_break_scenario"))
    if mb not in ["model_failure", "model_success"]:
        continue
    overall_counts[mb] += 1
    overall_counts["total_count"] += 1

overall_distribution = {}
if overall_counts["total_count"] > 0:
    overall_distribution = {
        "total_count": overall_counts["total_count"],
        "model_failure": {
            "count": overall_counts["model_failure"],
            "percentage": round((overall_counts["model_failure"] / overall_counts["total_count"]) * 100, 2)
        },
        "model_success": {
            "count": overall_counts["model_success"],
            "percentage": round((overall_counts["model_success"] / overall_counts["total_count"]) * 100, 2)
        }
    }

# Write overall distribution to JSON.
json_out5 = os.path.join(output_folder, 'overall_model_break_distribution.json')
with open(json_out5, 'w', encoding='utf-8') as f:
    json.dump(overall_distribution, f, indent=4)

print("Files generated successfully in folder:", output_folder)
