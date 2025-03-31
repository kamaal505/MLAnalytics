import json
import os
import matplotlib.pyplot as plt

# Ask for the JSON file path
file_path = input("Enter the path to the JSON file: ").strip()

# Load the JSON data
with open(file_path, "r") as file:
    data = json.load(file)

# Define a color palette
colors = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", 
    "#1ABC9C", "#E67E22", "#D35400", "#C0392B", "#7F8C8D"
]

# Output folder
output_folder = os.path.dirname(file_path)

# Generate a pie chart for each category
for category, error_types in data.items():
    labels = list(error_types.keys())
    values = list(error_types.values())

    # Create pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)],
        startangle=140, wedgeprops={'edgecolor': 'black'}
    )

    # Improve text visibility
    for text in texts + autotexts:
        text.set_fontsize(10)
        text.set_color("black")

    ax.set_title(f"{category} - Error Breakdown")

    # Save plot
    output_path = os.path.join(output_folder, f"{category.replace(' ', '_')}_pie.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Pie chart saved: {output_path}")
