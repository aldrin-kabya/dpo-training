import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import gc

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# Paths to your trained checkpoints
models_to_evaluate = {
    "Standard_DPO": "saves/llama-3.2-1b/full/dpo/checkpoint-112",
    "Random_DPO":   "saves/llama-3.2-1b/full/dpo_random_neg/checkpoint-112"
}

# The Reward Model used in the reference paper
reward_model_id = "sfairXC/FsfairX-LLaMA3-RM-v0.1"

# Output file
output_file = "evaluation_results.json"

def load_validation_prompts():
    """
    Loads unique prompts from HelpSteer2. 
    Tries the official validation split first. 
    If not found, falls back to the last 200 items of the training set.
    """
    print(f"Loading data from nvidia/HelpSteer2...")
    
    try:
        # Try loading the official validation split
        ds = load_dataset("nvidia/HelpSteer2", split="validation")
        print("Successfully loaded official validation split.")
    except Exception as e:
        print(f"Warning: Could not load 'validation' split ({e}).")
        print("Fallback: Loading last 200 samples from 'train' split as a proxy.")
        ds = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train[-200:]")

    # Extract prompts. We use a set to ensure uniqueness.
    prompts = list(set([row['prompt'] for row in ds]))
    prompts.sort()
    
    print(f"Loaded {len(prompts)} unique prompts for evaluation.")
    return prompts

def generate_responses(model_path, prompts):
    """Generates responses from the fine-tuned model."""
    print(f"\n--- Loading Generator: {model_path} ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device
        )
    except Exception as e:
        print(f"Failed to load model at {model_path}: {e}")
        return []

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    responses = []
    
    # Generation Config
    gen_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    print(f"Generating responses for {len(prompts)} prompts...")
    
    for prompt in tqdm(prompts):
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        # We handle the case where it returns a dict (BatchEncoding) or a Tensor
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        
        # Robustly extract input_ids
        if isinstance(inputs, dict) or hasattr(inputs, "keys"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
        else:
            input_ids = inputs
            attention_mask = None
            
        # Move to device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            # Pass input_ids explicitly as a keyword argument to avoid ambiguity
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        # Decode only the generated response
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(generated_text)

    # Clean up memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return responses

def compute_reward_scores(prompts, responses):
    """Scores the (prompt, response) pairs using the Reward Model."""
    print(f"\n--- Loading Reward Model: {reward_model_id} ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_id,
            torch_dtype=dtype,
            device_map=device,
            num_labels=1
        )
    except Exception as e:
        print(f"Error loading Reward Model: {e}")
        return []
    
    scores = []
    
    print("Scoring responses...")
    for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        # Apply chat template for RM
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        
        with torch.no_grad():
            output = rm_model(**inputs)
            score = output.logits[0].item()
            scores.append(score)
            
    del rm_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return scores

def main():
    prompts = load_validation_prompts()
    
    if not prompts:
        print("No prompts found. Exiting.")
        return

    # Optional: For testing, uncomment next line to limit to 5 prompts
    # prompts = prompts[:5]

    results = {}
    
    for model_name, path in models_to_evaluate.items():
        if not os.path.exists(path):
            print(f"Skipping {model_name}: Path not found.")
            continue
            
        # A. Generate
        responses = generate_responses(path, prompts)
        if not responses: 
            continue
            
        # B. Score
        scores = compute_reward_scores(prompts, responses)
        
        # C. Calculate Average
        avg_score = sum(scores) / len(scores)
        print(f"Model: {model_name} | Average Reward Score: {avg_score:.4f}")
        
        results[model_name] = {
            "average_score": avg_score,
            "details": [
                {"prompt": p, "response": r, "score": s}
                for p, r, s in zip(prompts, responses, scores)
            ]
        }
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    for model_name, data in results.items():
        print(f"{model_name:15}: {data['average_score']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()