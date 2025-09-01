from matplotlib import pyplot as plt
import json
import os
import fire


label2name = {
    "ppo": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.2_0.28": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo w/o rc": None,
    "ppo+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.2_0.28+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",  # best one seems
    "ppo_0.2_0.28+aggr": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.5_1.0": "0819-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.5-1.0-trT0.6",
    "ppo_0.5_1.0+aggr": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.5-1.0-trT0.6",  
    "ppo+direct": "0831-2-mix6-remove-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6", # preliminary
    "plmo_single_0.2_0.28": "0831-3-record_pa_reward-mix6-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_single+cons": "0826-1-mix6-max2depth-cons-0821-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6", # nearest one
    "plmo_single+aggr": "0831-1-mix6-max2depth-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo w/o rc": "0830-1-record_pa_reward-mix6-plmo-average-n8-rloo-3072-kl0.0-cl1.0-1e09-trT0.6",
    "plmo_avg": "checkpoints/ckpts/0901-1-record_pa_reward-mix6-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_avg_0.1_0.12": "checkpoints/ckpts/0901-1-record_pa_reward-mix6-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_avg+cons": "0828-2-mix6-max2depth-cons-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_sum": "0828-1-record_pa_reward-mix6-plmo-sum-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_sum_0.2_0.28": "0828-1-record_pa_reward-mix6-plmo-sum-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_sum+cons": "0827-1-mix6-max2depth-cons-plmo-sum-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6"
}
# ppo_cons, plmo_single+cons, plmo_avg+cons ppo+aggr+large

show_plmo = ["ppo_0.2_0.28", "plmo w/o rc", "plmo_single_0.2_0.28", "plmo_avg_0.1_0.12", "plmo_sum_0.2_0.28"] # "ppo_wo_rc", 
show_aggr_cons = ["ppo_0.2_0.28", "ppo+direct", "ppo+aggr", "ppo+aggr+large", "ppo+large", "ppo+cons"]
show_minif2f = ["ppo_0.2_0.28", "ppo+cons", "plmo_single+cons", "plmo_avg+cons"] # limitation part
show_entropy = ["plmo_single+cons", "ppo+cons", "ppo", "plmo_avg+cons"] # 


def main(log_fp="results/summary.log", output_root="results/paper_plot"):
    """
    Generate the ProofAug+ plot from the paper
    and save the pngs into output_root
    """
    
    os.makedirs(output_root, exist_ok=True)
    
    # Generate show_plmo plot
    generate_show_plmo_plot(log_fp, output_root)


def generate_show_plmo_plot(log_fp, output_root, max_step=72):
    """Generate plot for show_plmo comparison"""
    import numpy as np
    import ast
    
    # Read and parse data (Python dict format, not JSON)
    data = []
    with open(log_fp, 'r') as f:
        for line in f:
            try:
                # Use ast.literal_eval for Python dict format with single quotes
                item = ast.literal_eval(line.strip())
                data.append(item)
            except (ValueError, SyntaxError):
                # Skip malformed lines
                continue

    
    # Filter data for pset_test with -orig
    filtered_data = []
    for item in data:
        output_dir = item.get('output_dir', '')
        if 'pset_test' in output_dir and '-orig' in output_dir:
            filtered_data.append(item)
    
    # Group data by label
    results = {}
    
    # Process each label in show_plmo (excluding "plmo" which doesn't exist in label2name)
    labels_to_process = [label for label in show_plmo if label in label2name and label2name[label] is not None]
    
    for label in labels_to_process:
        model_path_pattern = label2name[label]
        
        # Find matching entries
        label_data = []
        for item in filtered_data:
            model_path = item.get('model', '')
            if model_path_pattern in model_path:
                # Extract global_step from model path (e.g., global_step2_hf)
                import re
                step_match = re.search(r'global_step(\d+)', model_path)
                if step_match:
                    step = int(step_match.group(1))
                    if step > max_step:
                        continue
                else:
                    # Skip if no global_step found
                    continue
                
                # Extract seed from output_dir (e.g., s5, s6, s7)
                output_dir = item.get('output_dir', '')
                seed_match = None
                for part in output_dir.split('-'):
                    if part.startswith('s') and part[1:].isdigit():
                        seed_match = int(part[1:])
                        break
                
                if seed_match is not None:
                    accuracy = float(item.get('accuracy', 0))
                    label_data.append({
                        'step': step,
                        'seed': seed_match,
                        'accuracy': accuracy,
                        'model': model_path,
                        'output_dir': output_dir
                    })
        
        if label_data:
            results[label] = label_data
    
    # Calculate mean and std for each label and step
    plot_data = {}
    for label, data_list in results.items():
        # Group by step
        step_groups = {}
        for item in data_list:
            step = item['step']
            if step not in step_groups:
                step_groups[step] = []
            step_groups[step].append(item['accuracy'])
        
        # Calculate statistics for each step
        steps = []
        means = []
        stds = []
        for step in sorted(step_groups.keys()):
            accuracies = step_groups[step]
            steps.append(step)
            means.append(np.mean(accuracies))
            stds.append(np.std(accuracies) if len(accuracies) > 1 else 0)
        
        plot_data[label] = {
            'steps': steps,
            'means': means,
            'stds': stds,
            'n_points': len(data_list)
        }
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (label, data) in enumerate(plot_data.items()):
        steps = data['steps']
        means = data['means']
        stds = data['stds']
        
        color = colors[i % len(colors)]
        
        # Plot line without error bars
        plt.plot(steps, means, marker='o', linewidth=2, markersize=6,
                label=label, color=color)
        
        # Fill between for error background (lighter color)
        means_array = np.array(means)
        stds_array = np.array(stds)
        plt.fill_between(steps, means_array - stds_array, means_array + stds_array, 
                        alpha=0.2, color=color)
    
    plt.xlabel('Epoch')
    plt.ylabel('Pass@1(%)')
    # plt.title('Training Progress: Pass@1 vs Training Steps on PSet Test (Original)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_root, 'show_plmo.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Show PLMO training curves saved to: {output_path}")
    
    return plot_data

if __name__ == "__main__":
    fire.Fire(main)