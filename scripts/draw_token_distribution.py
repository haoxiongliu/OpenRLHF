from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt
import numpy as np


def draw_token_distribution(record_path: str, tokenizer_path: str = "AI-MO/Kimina-Prover-Distill-1.7B", save_path: str = None, only_correct: bool = True):
    """
    Draw token count histogram of the records in the record_path
    like results/minif2f_test_kimina/hf_models/Kimina-Prover-Distill-1.7B-n1-8192-T1.0-s4-orig/full_records.jsonl
    from the model_outputs field
    
    Args:
        record_path: Path to the JSONL file containing records
        tokenizer_path: Path to the tokenizer model
        save_path: Optional path to save the histogram image
        only_correct: If True, only analyze correctly solved problems using code_compilation.json
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    token_counts = []
    
    # Load correctness information if needed
    correct_problems = set()
    if only_correct:
        import os
        # Get the directory containing the record_path
        record_dir = os.path.dirname(record_path)
        code_compilation_path = os.path.join(record_dir, "code_compilation.json")
        
        if os.path.exists(code_compilation_path):
            print(f"Loading correctness information from {code_compilation_path}...")
            with open(code_compilation_path, "r") as f:
                compilation_data = json.load(f)
                for entry in compilation_data:
                    result = entry.get("compilation_result", {})
                    # Check if the problem was solved correctly
                    if (result.get("rewards", 0) == 1.0 and 
                        result.get("complete", False) == True):
                        correct_problems.add(entry["name"])
            print(f"Found {len(correct_problems)} correctly solved problems")
        else:
            print(f"Warning: code_compilation.json not found at {code_compilation_path}")
            print("Analyzing all problems instead")
            only_correct = False
    
    print(f"Loading records from {record_path}...")
    processed_problems = 0
    skipped_problems = 0
    
    with open(record_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                problem_name = record.get("name", "")
                
                # Filter by correctness if requested
                if only_correct and problem_name not in correct_problems:
                    skipped_problems += 1
                    continue
                
                model_outputs = record["model_outputs"]
                for output in model_outputs:
                    tokens = tokenizer.encode(output)
                    token_counts.append(len(tokens))
                processed_problems += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Processed {processed_problems} problems, skipped {skipped_problems} problems")
    
    if not token_counts:
        print("No valid token counts found!")
        return
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    
    # Create histogram
    n_bins = min(50, max_tokens - min_tokens + 1)  # Reasonable number of bins
    plt.hist(token_counts, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(mean_tokens, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tokens:.1f}')
    plt.axvline(median_tokens, color='green', linestyle='--', linewidth=2, label=f'Median: {median_tokens:.1f}')
    
    # Formatting
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Update title based on filtering
    if only_correct and correct_problems:
        title = f'Distribution of Output Token Counts (Correct Solutions Only)\n(Total samples: {len(token_counts)}, Correct problems: {len(correct_problems)})'
    else:
        title = f'Distribution of Output Token Counts (All Problems)\n(Total samples: {len(token_counts)})'
    
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Statistics:\nMin: {min_tokens}\nMax: {max_tokens}\nMean: {mean_tokens:.1f}\nMedian: {median_tokens:.1f}\nStd: {np.std(token_counts):.1f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\nToken Distribution Summary:")
    if only_correct and correct_problems:
        print(f"Analysis mode: Correct solutions only")
        print(f"Correct problems found: {len(correct_problems)}")
        print(f"Problems processed: {processed_problems}")
        print(f"Problems skipped: {skipped_problems}")
    else:
        print(f"Analysis mode: All problems")
        print(f"Problems processed: {processed_problems}")
    print(f"Total token samples: {len(token_counts)}")
    print(f"Min tokens: {min_tokens}")
    print(f"Max tokens: {max_tokens}")
    print(f"Mean tokens: {mean_tokens:.1f}")
    print(f"Median tokens: {median_tokens:.1f}")
    print(f"Standard deviation: {np.std(token_counts):.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate token distribution histogram for model outputs")
    parser.add_argument("--record_path", type=str, default="results/minif2f_test_kimina/hf_models/Kimina-Prover-Distill-1.7B-n1-8192-T1.0-s4-orig/full_records.jsonl",
                        help="Path to the JSONL file containing model outputs")
    parser.add_argument("--tokenizer_path", type=str, default="AI-MO/Kimina-Prover-Distill-1.7B",
                        help="Path to the tokenizer model")
    parser.add_argument("--save_path", type=str, default="results/token_distribution.png",
                        help="Path to save the histogram image")
    parser.add_argument("--only_correct", action="store_true", default=True,
                        help="Only analyze correctly solved problems (default: True)")
    parser.add_argument("--all_problems", action="store_true", 
                        help="Analyze all problems regardless of correctness")
    args = parser.parse_args()
    
    # Handle the logic for only_correct vs all_problems
    only_correct = args.only_correct and not args.all_problems
    
    draw_token_distribution(args.record_path, args.tokenizer_path, args.save_path, only_correct)