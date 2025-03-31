import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Ask for the JSON file path
file_path = input("Enter the path to the JSON file: ").strip()

# Load the JSON data
with open(file_path, "r") as file:
    data = json.load(file)

# Extract relevant information
categories = list(data.keys())  # X-axis labels
success_rates = [data[cat]["model_success"] for cat in categories]
failure_rates = [data[cat]["model_failure"] for cat in categories]

# Define colors
success_color = "#2ECC71"  # Bright green
failure_color = "#E74C3C"  # Bright red

# Bar chart setup
x = np.arange(len(categories))
width = 0.6  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Plot stacked bars
ax.bar(x, failure_rates, width, label="Model Failure", color=failure_color)
ax.bar(x, success_rates, width, bottom=failure_rates, label="Model Success", color=success_color)

# Formatting
ax.set_xlabel("Prompt Type")
ax.set_ylabel("Percentage")
ax.set_title("Model Success and Failure by Prompt Type")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha="right")
ax.legend()

# Save plot in the same folder as the input file
output_path = os.path.join(os.path.dirname(file_path), "model_break_scenario_by_prompt_type.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved to: {output_path}")

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Ask for the JSON file path
file_path = input("Enter the path to the JSON file: ").strip()

# Load the JSON data
with open(file_path, "r") as file:
    data = json.load(file)

# Extract relevant information
categories = list(data.keys())  # X-axis labels
success_rates = [data[cat]["model_success"] for cat in categories]
failure_rates = [data[cat]["model_failure"] for cat in categories]

# Define colors
success_color = "#2ECC71"  # Bright green
failure_color = "#E74C3C"  # Bright red

# Bar chart setup
x = np.arange(len(categories))
width = 0.6  # Bar width

fig, ax = plt.subplots(figsize=(8, 5))

# Plot stacked bars
ax.bar(x, failure_rates, width, label="Model Failure", color=failure_color)
ax.bar(x, success_rates, width, bottom=failure_rates, label="Model Success", color=success_color)

# Formatting
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Percentage")
ax.set_title("Model Success and Failure by Difficulty Level")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha="center")
ax.legend()

# Save plot in the same folder as the input file
output_path = os.path.join(os.path.dirname(file_path), "probability_complexity_vs_model_break.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved to: {output_path}")
