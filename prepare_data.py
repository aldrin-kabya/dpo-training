import json
from datasets import load_dataset

def convert_helpsteer_to_dpo():
    print("Loading HelpSteer2 preference dataset...")
    # Load the preference subset of HelpSteer2
    ds = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")
    
    output_data = []
    
    print("Processing rows...")
    for row in ds:
        strength = row['preference_strength']
        
        # Skip ties (strength 0) as DPO needs a clear winner
        if strength == 0:
            continue
        
        prompt = row['prompt']
        r1 = row['response_1']
        r2 = row['response_2']
        
        # Determine chosen and rejected based on preference strength
        # Positive strength (>0) means Response 2 is better
        # Negative strength (<0) means Response 1 is better
        if strength > 0:
            chosen = r2
            rejected = r1
        else:
            chosen = r1
            rejected = r2
            
        output_data.append({
            "instruction": prompt,
            "input": "",
            "chosen": chosen,
            "rejected": rejected
        })
        
    # Save to data directory
    output_file = "data/helpsteer2_dpo.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(output_data)} samples to {output_file}")

if __name__ == "__main__":
    convert_helpsteer_to_dpo()