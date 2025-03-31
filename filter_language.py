import json
import os
import re
import csv

def contains_chinese(text):
    """Check if the given text contains any Chinese characters."""
    return bool(re.search(r'[一-鿿]', text))

def count_model_breaks(conversations):
    """Count the number of model break instances in a given list of conversations."""
    return sum(
        1 for conversation in conversations for evaluation in conversation.get("modelEvaluations", []) if evaluation.get("model break") == "True"
    )

def filter_conversations(input_file):
    """Remove conversations where any modelResponse contains Chinese characters and save model break prompts."""
    try:
        # Load JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the data is a list
        if not isinstance(data, list):
            raise ValueError("JSON root should be a list of conversations.")
        
        input_count = len(data)
        removed_conversations = []
        filtered_data = []
        model_break_prompts = []
        
        # Count total model break instances in input data
        total_model_breaks_input = count_model_breaks(data)
        
        # Process conversations
        for conversation in data:
            model_responses = conversation.get("modelResponses", [])
            if any(contains_chinese(resp.get("modelResponse", "")) for resp in model_responses):
                removed_conversations.append(conversation)
            else:
                filtered_data.append(conversation)
        
        output_count = len(filtered_data)
        
        # Count total model break instances in filtered data and removed conversations
        total_model_breaks_filtered = count_model_breaks(filtered_data)
        total_model_breaks_removed = count_model_breaks(removed_conversations)
        
        # Extract model break prompts from removed conversations
        for conversation in removed_conversations:
            user_prompt = conversation.get("userPrompt", "")
            final_answer = conversation.get("finalAnswer", "")
            model_evaluations = conversation.get("modelEvaluations", [])
            
            for evaluation in model_evaluations:
                if evaluation.get("model break") == "True":
                    model_break_prompts.append([conversation.get("conversationId", "Unknown"), user_prompt, final_answer])
        
        model_break_count = len(model_break_prompts)
        
        # Determine output file paths
        output_file = os.path.join(os.path.dirname(input_file), "filtered_batch.json")
        csv_file = os.path.join(os.path.dirname(input_file), "model_break_prompts.csv")
        
        # Save filtered data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        
        # Save model break prompts to CSV
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Conversation ID", "User Prompt", "Final Answer"])
            writer.writerows(model_break_prompts)
        
        print(f"Filtered data saved to: {output_file}")
        print(f"Model break prompts saved to: {csv_file}")
        print(f"Number of conversations in input file: {input_count}")
        print(f"Number of conversations in output file: {output_count}")
        print(f"Total model break scenarios in input JSON: {total_model_breaks_input}")
        print(f"Total model break scenarios in filtered JSON: {total_model_breaks_filtered}")
        print(f"Total model break scenarios in removed conversations: {total_model_breaks_removed}")
        print(f"Total model break scenarios saved in CSV: {model_break_count}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    input_path = input("Enter the JSON file path: ").strip()
    if os.path.exists(input_path) and input_path.endswith(".json"):
        filter_conversations(input_path)
    else:
        print("Invalid file path. Please provide a valid JSON file.")
