"""
Draw histograms of the tactic usage statistics.
Output the top tactic usage rate (and the sum of top 10).
To figure out if 'tactic-overfitting' really happen for ProofAug+RL.
results/pset_test/0812-q2515bi-pset10k-sft-pset140k-n8-rloo-3090-bs64-mix6-remove-0809-3072-kl0.0-ch0.3/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl
"""

import re
from prover.utils import PROOF_PATTERN, load_jsonl_objects
import fire
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_line_tactic(line: str) -> list[str|None]:
    """
    Extract the tactic from the line.
    """
    line.lstrip()
    words = line.split(" ")
    components = []
    for i, word in enumerate(words):
        if word == "<;>" and i == 0:
            continue
        elif word == "try" or ";" in word:
            continue
        else:
            components.append(word)
            break
    
    return " ".join(components)

def create_separate_distributions(all_tactic_frequencies, record_labels, top_n=20):
    """
    Create separate histogram distributions for each dataset.
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate plots for each dataset
    for i, (file_idx, frequencies) in enumerate(all_tactic_frequencies.items()):
        # Get top tactics for this dataset
        top_tactics = frequencies[:top_n]
        tactics = [tactic for tactic, _ in top_tactics]
        counts = [freq for _, freq in top_tactics]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar plot without specific tactic names on x-axis
        x_positions = range(len(tactics))
        bars = plt.bar(x_positions, counts, alpha=0.7, color=f'C{i}')
        
        # Set labels and title
        plt.xlabel('Tactic Rank (Most to Least Frequent)')
        plt.ylabel('Count')
        plt.title(f'Tactic Count Distribution - {record_labels[file_idx]}')
        
        # Remove specific tactic names from x-axis, just show rank numbers
        plt.xticks(x_positions[::2], [str(i+1) for i in x_positions[::2]])  # Show every 2nd rank
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add some statistics as text
        total_tactics = sum(counts)
        top_5_percent = sum(counts[:5]) / total_tactics * 100
        plt.text(0.02, 0.98, f'Total tactics: {total_tactics}\nTop 5 tactics: {top_5_percent:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Tight layout
        plt.tight_layout()
        
        # Save individual plot
        output_path = os.path.join(output_dir, f"tactic_distribution_{record_labels[file_idx]}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution saved to: {output_path}")
        
        plt.show()
        plt.close()
    
    # Also create a summary comparison of distribution shapes
    create_distribution_comparison(all_tactic_frequencies, record_labels, top_n)

def create_distribution_comparison(all_tactic_frequencies, record_labels, top_n=20):
    """
    Create a line plot comparing the distribution shapes.
    """
    plt.figure(figsize=(12, 8))
    
    for i, (file_idx, frequencies) in enumerate(all_tactic_frequencies.items()):
        # Get raw counts instead of percentages
        top_tactics = frequencies[:top_n]
        counts = [freq for _, freq in top_tactics]
        
        # Plot distribution shape
        x_positions = range(1, len(counts) + 1)
        plt.plot(x_positions, counts, marker='o', linewidth=2, 
                label=record_labels[file_idx], alpha=0.8)
    
    plt.xlabel('Tactic Rank')
    plt.ylabel('Count')
    # plt.title('Tactic Count Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better show the distribution
    
    # Save comparison plot
    output_dir = "results"
    output_path = os.path.join(output_dir, "tactic_distribution_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison saved to: {output_path}")
    
    plt.show()
    plt.close()
    
def main(record_fps=None, record_labels=None):
    if record_fps is None:
        record_fps = ["results/pset_test/0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl", 
                    #   "results/pset_test/0831-2-mix6-remove-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl",
                      "results/pset_test/0812-q2515bi-pset10k-sft-pset140k-n8-rloo-3090-bs64-mix6-remove-0809-3072-kl0.0-ch0.3/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl", 
                      "results/pset_test/0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl", 
                      "results/pset_test/0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.5-1.0-trT0.6/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl"]
        record_labels = ["original_step10", "direct_step10", "conservative_step10", "aggressive_step10"]
        # "results/pset_test/0826-1-mix6-max2depth-cons-0821-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl"
    
    if record_labels is None:
        record_labels = [f"Dataset_{i+1}" for i in range(len(record_fps))]
    
    all_tactic_frequencies = {}
    
    # Process each file
    for i, record_fp in enumerate(record_fps):
        tactic2freq = defaultdict(int)
        data = load_jsonl_objects(record_fp)
        
        for item in data:
            full_code = item["full_code"][0]
            full_code = "" if not full_code else full_code
            if not re.match(PROOF_PATTERN, full_code):
                continue
            m = re.match(PROOF_PATTERN, full_code)
            if not m:
                continue
            body = m.group('suffix')
            lines = body.strip().split("\n")
            for line in lines:
                tactic = extract_line_tactic(line)
                if tactic and tactic != "have":  # Exclude "have" tactic
                    tactic2freq[tactic] += 1
    
        tactic2freq = sorted(tactic2freq.items(), key=lambda x: x[1], reverse=True)
        all_tactic_frequencies[i] = tactic2freq
        
        # Print top tactics for this file
        print(f"\nTop tactics for {record_labels[i]}:")
        for tactic, freq in tactic2freq[:10]:
            print(f"  {tactic}: {freq}")
    
    # Create separate distribution histograms
    create_separate_distributions(all_tactic_frequencies, record_labels)

if __name__ == "__main__":
    fire.Fire(main)