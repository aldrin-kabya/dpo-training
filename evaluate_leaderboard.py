import lm_eval
import torch
import json
import os
import gc

# --- Configuration ---
# Tasks matches Table 6 of the paper
tasks_to_run = [
    "arc_easy",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande"
]

# Paths to your trained checkpoints
models_to_evaluate = {
    "Standard_DPO": "saves/llama-3.2-1b/full/dpo/checkpoint-112",
    "Random_DPO":   "saves/llama-3.2-1b/full/dpo_random_neg/checkpoint-112"
}

output_file = "leaderboard_results.json"
device = "cuda"

def get_metric_value(metrics_dict, preferred_metrics):
    """
    Robustly extracts values from lm-eval output.
    """
    for metric in preferred_metrics:
        # 1. Check for exact match
        if metric in metrics_dict:
            return float(metrics_dict[metric])
        
        # 2. Check for key with suffix (e.g., 'acc,none' or 'acc,stderr')
        for key in metrics_dict.keys():
            if key.startswith(metric + ","):
                return float(metrics_dict[key])
    return 0.0

def run_evaluation(model_name, model_path):
    print(f"\n" + "="*50)
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print("="*50)

    # Load model
    model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks_to_run,
        device=device,
        batch_size="auto"
    )

    return results

def main():
    final_results = {}

    for name, path in models_to_evaluate.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: Path {path} not found.")
            continue
            
        try:
            # Run Eval
            eval_output = run_evaluation(name, path)
            
            final_results[name] = {}
            print(f"\n--- Results for {name} ---")
            
            results_dict = eval_output["results"]
            mmlu_scores = []
            
            # 1. Extract Scores per Task
            for task_key, metrics in results_dict.items():
                # Define metric priority
                if "gsm8k" in task_key:
                    target_metrics = ["exact_match", "q_exact_match"]
                elif "truthfulqa" in task_key:
                    target_metrics = ["acc"] # MC2
                else:
                    target_metrics = ["acc_norm", "acc"] # Default for others

                score = get_metric_value(metrics, target_metrics)
                normalized_score = score * 100
                
                # MMLU is split into many subtasks, we aggregate them later
                if "mmlu" in task_key:
                    mmlu_scores.append(normalized_score)
                else:
                    # Save individual task scores directly
                    # Map task names to cleaner keys if needed
                    clean_name = task_key
                    final_results[name][clean_name] = normalized_score

            # 2. Compute MMLU Average
            if mmlu_scores:
                mmlu_avg = sum(mmlu_scores) / len(mmlu_scores)
                final_results[name]["mmlu"] = mmlu_avg
            else:
                final_results[name]["mmlu"] = 0.0

            # 3. Compute OLL Average (The single score from the paper)
            # We look for the 6 specific keys. 
            # Note: lm-eval task keys might differ slightly, so we map them safely.
            
            # Helper to find score in our results dict
            def find_score(substring):
                for k, v in final_results[name].items():
                    if substring in k:
                        return v
                return 0.0

            s_arc = find_score("arc_easy")
            s_gsm = find_score("gsm8k")
            s_hel = find_score("hellaswag")
            s_mml = final_results[name]["mmlu"]
            s_tru = find_score("truthfulqa")
            s_win = find_score("winogrande")

            oll_average = (s_arc + s_gsm + s_hel + s_mml + s_tru + s_win) / 6.0
            final_results[name]["OLL_Average"] = oll_average

            print(f"OLL Average Score: {oll_average:.2f}")

            # Clean up
            del eval_output
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nEvaluation Complete. Detailed results saved to {output_file}")
    
    # Print Final Comparison Table
    print("\n" + "="*80)
    print(f"{'Task':<30} | {'Standard_DPO':<15} | {'Random_DPO':<15}")
    print("-" * 80)
    
    display_keys = [
        ("ARC-Easy", "arc_easy"), 
        ("GSM8K", "gsm8k"), 
        ("Hellaswag", "hellaswag"), 
        ("MMLU", "mmlu"), 
        ("TruthfulQA", "truthfulqa"), 
        ("Winogrande", "winogrande"),
        (">>> OLL Average", "OLL_Average")
    ]
    
    for label, key_substr in display_keys:
        # Helper to fetch score safely from nested dict
        def get_val(model):
            res = final_results.get(model, {})
            # Direct match
            if key_substr in res: return res[key_substr]
            # Partial match
            for k, v in res.items():
                if key_substr in k: return v
            return 0.0

        val_std = get_val("Standard_DPO")
        val_rnd = get_val("Random_DPO")
        
        print(f"{label:<30} | {val_std:<15.2f} | {val_rnd:<15.2f}")
    print("="*80)

if __name__ == "__main__":
    main()