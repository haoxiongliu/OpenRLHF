from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt
import numpy as np
import re


def calculate_max_indent(text: str) -> int:
    """
    Calculate the maximum indentation (number of leading spaces) in the given text
    """
    lines = text.split('\n')
    max_indent = 0
    
    for line in lines:
        # Only count lines that are not empty
        if line.strip():
            # Count leading spaces
            indent = len(line) - len(line.lstrip(' '))
            max_indent = max(max_indent, indent)
    
    return max_indent


def analyze_outliers(data, name="data"):
    """
    Analyze outliers in the data using IQR method
    """
    if not data:
        return {}
    
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    
    normal_values = [v for v in data if v <= outlier_threshold]
    outliers = [v for v in data if v > outlier_threshold]
    
    return {
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'outlier_threshold': outlier_threshold,
        'normal_values': normal_values,
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(data) * 100 if data else 0,
        'outlier_unique': sorted(set(outliers)) if outliers else []
    }


def draw_comprehensive_statistics(record_path: str, tokenizer_path: str = "AI-MO/Kimina-Prover-Distill-1.7B", save_path: str = None, only_correct: bool = True):
    """
    Draw comprehensive statistics including token count, max indent, and verify times
    from the records in the record_path
    
    Args:
        record_path: Path to the JSONL file containing records
        tokenizer_path: Path to the tokenizer model
        save_path: Optional path to save the histogram image
        only_correct: If True, only analyze correctly solved problems using code_compilation.json
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    token_counts = []
    max_indents = []
    verify_times = []
    
    # Load correctness information and verify times from code_compilation.json
    correct_problems = set()
    problem_verify_times = {}
    
    import os
    # Get the directory containing the record_path
    record_dir = os.path.dirname(record_path)
    code_compilation_path = os.path.join(record_dir, "code_compilation.json")
    
    if os.path.exists(code_compilation_path):
        print(f"Loading compilation information from {code_compilation_path}...")
        with open(code_compilation_path, "r") as f:
            compilation_data = json.load(f)
            for entry in compilation_data:
                problem_name = entry.get("name", "")
                result = entry.get("compilation_result", {})
                
                # Extract verify times
                verify_count = entry.get("verify_times", result.get("verify_times", result.get("num_verifications", result.get("verification_count", 1))))
                problem_verify_times[problem_name] = verify_count
                
                # Check if the problem was solved correctly
                if only_correct:
                    if (result.get("rewards", 0) == 1.0 and 
                        result.get("complete", False) == True):
                        correct_problems.add(problem_name)
        
        if only_correct:
            print(f"Found {len(correct_problems)} correctly solved problems")
        print(f"Found verify_times data for {len(problem_verify_times)} problems")
    else:
        print(f"Warning: code_compilation.json not found at {code_compilation_path}")
        if only_correct:
            print("Analyzing all problems instead")
            only_correct = False
        print("No verify_times data available")
    
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
                
                # Extract verify_times from code_compilation data
                verify_count = problem_verify_times.get(problem_name, 1)  # Default to 1 if not found
                verify_times.append(verify_count)
                
                for output in model_outputs:
                    # Token count
                    tokens = tokenizer.encode(output)
                    token_counts.append(len(tokens))
                    
                    # Max indent
                    max_indent = calculate_max_indent(output)
                    max_indents.append(max_indent)
                
                processed_problems += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Processed {processed_problems} problems, skipped {skipped_problems} problems")
    
    if not token_counts:
        print("No valid data found!")
        return
    
    # Create subplots for comprehensive statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Token distribution
    ax1 = axes[0, 0]
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    n_bins = int(min(50, max(token_counts) - min(token_counts) + 1))
    ax1.hist(token_counts, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(mean_tokens, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tokens:.1f}')
    ax1.axvline(median_tokens, color='green', linestyle='--', linewidth=2, label=f'Median: {median_tokens:.1f}')
    ax1.set_xlabel('Number of Tokens')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Token Count Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Max indent distribution
    ax2 = axes[0, 1]
    mean_indent = np.mean(max_indents)
    median_indent = np.median(max_indents)
    n_bins_indent = int(min(30, max(max_indents) - min(max_indents) + 1)) if max_indents else 1
    ax2.hist(max_indents, bins=n_bins_indent, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(mean_indent, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_indent:.1f}')
    ax2.axvline(median_indent, color='green', linestyle='--', linewidth=2, label=f'Median: {median_indent:.1f}')
    ax2.set_xlabel('Maximum Indentation (spaces)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Maximum Indentation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Verify times distribution with outlier handling
    ax3 = axes[1, 0]
    if verify_times and any(v > 0 for v in verify_times):
        mean_verify = np.mean(verify_times)
        median_verify = np.median(verify_times)
        
        # Analyze outliers
        outlier_analysis = analyze_outliers(verify_times, "verify_times")
        normal_values = outlier_analysis['normal_values']
        outliers = outlier_analysis['outliers']
        outlier_threshold = outlier_analysis['outlier_threshold']
        
        # Create histogram for normal values with finer binning for low values
        if normal_values:
            max_normal = max(normal_values)
            min_normal = min(normal_values)
            
            # Create finer bins for better resolution in time values
            if max_normal <= 2:
                # For very small times (0-2s), use bins of width 0.05s for fine detail
                bins = np.arange(min_normal - 0.025, max_normal + 0.075, 0.05)
            elif max_normal <= 5:
                # For small times (0-5s), use bins of width 0.1s
                bins = np.arange(min_normal - 0.05, max_normal + 0.15, 0.1)
            elif max_normal <= 10:
                # For moderate times (0-10s), use bins of width 0.2s
                bins = np.arange(min_normal - 0.1, max_normal + 0.3, 0.2)
            elif max_normal <= 20:
                # For larger times (0-20s), use bins of width 0.5s
                bins = np.arange(min_normal - 0.25, max_normal + 0.75, 0.5)
            else:
                # For very large times, use adaptive binning
                n_bins_normal = int(min(30, (max_normal - min_normal) / 1.0 + 1))
                bins = n_bins_normal
                
            ax3.hist(normal_values, bins=bins, alpha=0.7, color='lightcoral', 
                    edgecolor='black', label=f'Normal values (≤{outlier_threshold:.1f}s)')
        
        # Add outliers as text annotations directly on the plot
        if outliers:
            # Create outlier summary text
            outlier_counts = {}
            for outlier in outliers:
                outlier_counts[outlier] = outlier_counts.get(outlier, 0) + 1
            
            # Sort outliers by value
            sorted_outliers = sorted(outlier_counts.items())
            
            # Create outlier text summary
            outlier_text_lines = []
            if len(sorted_outliers) <= 8:  # Show all if not too many
                for value, count in sorted_outliers:
                    if count == 1:
                        outlier_text_lines.append(f'{value:.2f}s')
                    else:
                        outlier_text_lines.append(f'{value:.2f}s×{count}')
            else:  # Show top 6 and summary
                for value, count in sorted_outliers[:6]:
                    if count == 1:
                        outlier_text_lines.append(f'{value:.2f}s')
                    else:
                        outlier_text_lines.append(f'{value:.2f}s×{count}')
                remaining_count = sum(count for _, count in sorted_outliers[6:])
                outlier_text_lines.append(f'...+{remaining_count} more')
            
            # Add outlier text box
            outlier_text = f'Outliers (>{outlier_threshold:.1f}s):\n' + ', '.join(outlier_text_lines)
            ax3.text(0.98, 0.70, outlier_text, transform=ax3.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), 
                    fontsize=9, color='darkred')
        
        # Add statistical lines
        ax3.axvline(mean_verify, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_verify:.1f}')
        ax3.axvline(median_verify, color='green', linestyle='--', linewidth=2, label=f'Median: {median_verify:.1f}')
        
        # Add percentile information
        ax3.axvline(outlier_analysis['q75'], color='orange', linestyle=':', linewidth=1, alpha=0.7, 
                   label=f'Q3: {outlier_analysis["q75"]:.1f}')
        
        ax3.set_xlabel('Verification Time (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Verification Time Distribution\n(from code_compilation.json)')
        ax3.legend(fontsize=8, loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Add comprehensive stats text (moved to avoid overlap with outlier text)
        stats_text = f'Outliers: {outlier_analysis["outlier_count"]}/{len(verify_times)} ({outlier_analysis["outlier_percentage"]:.1f}%)\nIQR: {outlier_analysis["iqr"]:.2f}s\nThreshold: >{outlier_threshold:.1f}s'
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No verification data available\nfrom code_compilation.json', transform=ax3.transAxes,
                ha='center', va='center', fontsize=12)
        ax3.set_title('Verification Time Distribution\n(from code_compilation.json)')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""Summary Statistics:

Tokens:
  • Samples: {len(token_counts)}
  • Min: {min(token_counts)}
  • Max: {max(token_counts)}
  • Mean: {mean_tokens:.1f}
  • Median: {median_tokens:.1f}
  • Std: {np.std(token_counts):.1f}

Max Indentation:
  • Min: {min(max_indents)}
  • Max: {max(max_indents)}
  • Mean: {mean_indent:.1f}
  • Median: {median_indent:.1f}
  • Std: {np.std(max_indents):.1f}"""
    
    if verify_times and any(v > 0 for v in verify_times):
        outlier_analysis = analyze_outliers(verify_times, "verify_times")
        summary_text += f"""

Verification Times (from code_compilation):
  • Min: {min(verify_times):.3f}s
  • Max: {max(verify_times):.3f}s
  • Mean: {np.mean(verify_times):.3f}s
  • Median: {np.median(verify_times):.3f}s
  • Q3: {outlier_analysis['q75']:.3f}s
  • IQR: {outlier_analysis['iqr']:.3f}s
  • Outliers: {outlier_analysis['outlier_count']} ({outlier_analysis['outlier_percentage']:.1f}%)"""
    
    if only_correct and correct_problems:
        summary_text += f"""

Analysis Mode: Correct solutions only
Correct problems: {len(correct_problems)}
Problems processed: {processed_problems}
Problems skipped: {skipped_problems}"""
    else:
        summary_text += f"""

Analysis Mode: All problems
Problems processed: {processed_problems}"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive statistics saved to {save_path}")
    else:
        plt.show()
    
    # Print detailed summary
    print(f"\nComprehensive Statistics Summary:")
    if only_correct and correct_problems:
        print(f"Analysis mode: Correct solutions only")
        print(f"Correct problems found: {len(correct_problems)}")
        print(f"Problems processed: {processed_problems}")
        print(f"Problems skipped: {skipped_problems}")
    else:
        print(f"Analysis mode: All problems")
        print(f"Problems processed: {processed_problems}")
    
    print(f"\nToken Statistics:")
    print(f"  Total samples: {len(token_counts)}")
    print(f"  Min tokens: {min(token_counts)}")
    print(f"  Max tokens: {max(token_counts)}")
    print(f"  Mean tokens: {mean_tokens:.1f}")
    print(f"  Median tokens: {median_tokens:.1f}")
    print(f"  Standard deviation: {np.std(token_counts):.1f}")
    
    print(f"\nIndentation Statistics:")
    print(f"  Min max indent: {min(max_indents)}")
    print(f"  Max max indent: {max(max_indents)}")
    print(f"  Mean max indent: {mean_indent:.1f}")
    print(f"  Median max indent: {median_indent:.1f}")
    print(f"  Standard deviation: {np.std(max_indents):.1f}")
    
    if verify_times and any(v > 0 for v in verify_times):
        outlier_analysis = analyze_outliers(verify_times, "verify_times")
        print(f"\nVerification Time Statistics (from code_compilation.json):")
        print(f"  Min verify time: {min(verify_times):.3f}s")
        print(f"  Max verify time: {max(verify_times):.3f}s")
        print(f"  Mean verify time: {np.mean(verify_times):.3f}s")
        print(f"  Median verify time: {np.median(verify_times):.3f}s")
        print(f"  Standard deviation: {np.std(verify_times):.3f}s")
        print(f"  Q1 (25th percentile): {outlier_analysis['q25']:.3f}s")
        print(f"  Q3 (75th percentile): {outlier_analysis['q75']:.3f}s")
        print(f"  IQR: {outlier_analysis['iqr']:.3f}s")
        print(f"  Outlier threshold: >{outlier_analysis['outlier_threshold']:.1f}s")
        print(f"  Outliers: {outlier_analysis['outlier_count']}/{len(verify_times)} ({outlier_analysis['outlier_percentage']:.1f}%)")
        if outlier_analysis['outlier_unique']:
            print(f"  Unique outlier values (seconds): {[f'{v:.2f}' for v in outlier_analysis['outlier_unique']]}")
    else:
        print(f"\nVerification Time Statistics: No data available from code_compilation.json")


# Keep the original function for backward compatibility
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
    n_bins = int(min(50, max_tokens - min_tokens + 1))  # Reasonable number of bins, ensure integer
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
    parser = argparse.ArgumentParser(description="Generate comprehensive statistics for model outputs")
    parser.add_argument("--record_path", type=str, default="results/minif2f_test_kimina/hf_models/Kimina-Prover-Distill-1.7B-n1-8192-T1.0-s4-orig/full_records.jsonl",
                        help="Path to the JSONL file containing model outputs")
    parser.add_argument("--tokenizer_path", type=str, default="AI-MO/Kimina-Prover-Distill-1.7B",
                        help="Path to the tokenizer model")
    parser.add_argument("--save_path", type=str, default="results/comprehensive_statistics.png",
                        help="Path to save the statistics image")
    parser.add_argument("--only_correct", action="store_true", default=True,
                        help="Only analyze correctly solved problems (default: True)")
    parser.add_argument("--all_problems", action="store_true", 
                        help="Analyze all problems regardless of correctness")
    parser.add_argument("--comprehensive", action="store_true", default=True,
                        help="Generate comprehensive statistics including indent and verify times")
    parser.add_argument("--token_only", action="store_true",
                        help="Generate only token distribution (original functionality)")
    parser.add_argument("--log_scale", action="store_true",
                        help="Use logarithmic scale for better outlier visualization")
    
    args = parser.parse_args()
    
    # Handle the logic for only_correct vs all_problems
    only_correct = args.only_correct and not args.all_problems
    
    # Choose which function to call
    if args.token_only:
        draw_token_distribution(args.record_path, args.tokenizer_path, args.save_path, only_correct)
    else:
        draw_comprehensive_statistics(args.record_path, args.tokenizer_path, args.save_path, only_correct)