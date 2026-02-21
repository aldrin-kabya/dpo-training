import json
import random
from datasets import load_dataset

def prepare_random_mismatch():
    print("Loading HelpSteer2 preference dataset...")
    # Load the dataset
    ds = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")
    
    # 1. First pass: Collect all "Chosen" responses into a pool
    all_chosen_responses = []
    
    print("Collecting valid chosen responses...")
    valid_rows = []
    
    for row in ds:
        strength = row['preference_strength']
        
        # Skip ties
        if strength == 0:
            continue
            
        # Determine the winner
        if strength > 0:
            chosen_text = row['response_2']
        else:
            chosen_text = row['response_1']
            
        all_chosen_responses.append(chosen_text)
        
        # Save the row data + the determined chosen text for the next step
        valid_rows.append({
            "prompt": row['prompt'],
            "chosen": chosen_text
        })

    print(f"Collected {len(all_chosen_responses)} high-quality responses.")

    # 2. Second pass: Assign a random mismatch as the "Rejected" response
    output_data = []
    
    print("Generating mismatched pairs...")
    for item in valid_rows:
        prompt = item['prompt']
        chosen = item['chosen']
        
        # Pick a random response from the pool
        # Loop ensures we don't accidentally pick the exact same response
        while True:
            random_rejected = random.choice(all_chosen_responses)
            if random_rejected != chosen:
                break
        
        output_data.append({
            "instruction": prompt,
            "input": "",
            "chosen": chosen,
            "rejected": random_rejected  # This is the mismatch
        })
        
    # Save to data directory
    output_file = "data/helpsteer2_random_negatives.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(output_data)} samples to {output_file}")

if __name__ == "__main__":
    prepare_random_mismatch()