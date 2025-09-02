from matplotlib import pyplot as plt
import json
import os
import fire
import numpy as np
import ast
import re
from tensorboard.backend.event_processing import event_accumulator
import glob

label2name = {
    "gspo": "0901-3-record_pa_reward-mix6-gspo-n8-rloo-3072-kl0.0-cl0.2-0.27-trT0.6",
    "ppo": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.2_0.28": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.2_0.28+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",  # best one seems
    "ppo_0.2_0.28+aggr": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.5_1.0": "0819-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.5-1.0-trT0.6",
    "ppo_0.5_1.0+aggr": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.5-1.0-trT0.6",  
    "ppo_0.2_0.28+direct": "0831-2-mix6-remove-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6", # preliminary
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

show_plmo = ["gspo", "ppo_0.2_0.28", "plmo w/o rc", "plmo_single_0.2_0.28", "plmo_avg_0.1_0.12", "plmo_sum_0.2_0.28"] # "ppo_wo_rc", 
show_aggr_cons = ["ppo_0.2_0.28", "ppo_0.2_0.28+direct", "ppo_0.2_0.28+aggr", "ppo_0.5_1.0+aggr", "ppo_0.2_0.28+cons"] # "ppo_0.5_1.0", 
show_minif2f = ["ppo", "ppo+cons", "plmo_single+cons", "plmo_avg+cons"] # limitation part
show_entropy = ["plmo_single+cons", "ppo+cons", "ppo", "plmo_avg+cons"] # 


def main(log_fp="results/summary.log", output_root="results/paper_plot"):
    """
    Generate the ProofAug+ plot from the paper
    and save the pngs into output_root
    """
    
    os.makedirs(output_root, exist_ok=True)
    
    # Generate show_plmo plot
    generate_show_plmo_plot(log_fp, output_root, max_step=72)
    
    # Generate show_aggr_cons plot
    generate_show_aggr_cons_plot(log_fp, output_root, max_step=80)
    
    # Generate show_entropy plot
    generate_show_entropy_plot(log_fp, output_root, max_step=74)
    
    # Generate show_minif2f plot
    generate_show_minif2f_plot(log_fp, output_root, max_step=74)


def generate_training_curves_plot(log_fp, output_root, labels_to_show, output_filename, max_step=300, plot_title=None, dataset_filter=None):
    """
    Generic function to generate training curves plot for given labels
    
    Args:
        log_fp: Path to the log file
        output_root: Directory to save the plot
        labels_to_show: List of labels to include in the plot
        output_filename: Name of the output PNG file (without extension)
        max_step: Maximum training step to include
        plot_title: Optional title for the plot
        dataset_filter: Optional filter for dataset type (e.g., 'minif2f_test', 'pset_test')
    """

    
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

    
    # Filter data based on dataset_filter parameter
    filtered_data = []
    for item in data:
        output_dir = item.get('output_dir', '')
        if dataset_filter:
            # Use specific dataset filter
            if dataset_filter in output_dir and '-orig' in output_dir:
                filtered_data.append(item)
        else:
            # Default to pset_test for backward compatibility
            if 'pset_test' in output_dir and '-orig' in output_dir:
                filtered_data.append(item)
    
    # Group data by label
    results = {}
    
    # Process each label in labels_to_show
    labels_to_process = [label for label in labels_to_show if label in label2name and label2name[label] is not None]
    
    for label in labels_to_process:
        model_path_pattern = label2name[label]
        
        # Find matching entries
        label_data = []
        for item in filtered_data:
            model_path = item.get('model', '')
            if model_path_pattern in model_path:
                # Extract global_step from model path (e.g., global_step2_hf)
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
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
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
    # if plot_title:
    #     plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_root, f'{output_filename}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_path}")
    
    return plot_data


def generate_show_plmo_plot(log_fp, output_root, max_step=72):
    """Generate plot for show_plmo comparison"""
    return generate_training_curves_plot(
        log_fp=log_fp,
        output_root=output_root,
        labels_to_show=show_plmo,
        output_filename='show_plmo',
        max_step=max_step
    )


def generate_show_aggr_cons_plot(log_fp, output_root, max_step=72):
    """Generate plot for show_aggr_cons comparison"""
    return generate_training_curves_plot(
        log_fp=log_fp,
        output_root=output_root,
        labels_to_show=show_aggr_cons,
        output_filename='show_aggr_cons',
        max_step=max_step
    )


def load_entropy_data(tensorboard_dir, max_step=72):
    """Load entropy loss data from tensorboard logs"""
    entropy_data = {}
    
    # Find all event files in the directory
    event_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
    
    if not event_files:
        return entropy_data
    
    # Use the most recent event file
    event_file = max(event_files, key=os.path.getmtime)
    
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Check if entropy_loss exists
        if "train/entropy_loss" in ea.Tags()["scalars"]:
            scalars = ea.Scalars("train/entropy_loss")
            
            for scalar in scalars:
                step = scalar.step
                if step <= max_step:
                    entropy_data[step] = scalar.value
                    
    except Exception as e:
        print(f"Error loading entropy data from {event_file}: {e}")
    
    return entropy_data


def generate_show_entropy_plot(log_fp, output_root, max_step=72):
    """Generate plot for show_entropy comparison with dual y-axis for entropy"""
    
    # Get accuracy data using existing function
    accuracy_data = generate_training_curves_plot(
        log_fp=log_fp,
        output_root=output_root,
        labels_to_show=show_entropy,
        output_filename='temp_entropy',  # temporary file
        max_step=max_step
    )
    
    # Load entropy data for each label
    entropy_data = {}
    for label in show_entropy:
        if label in label2name:
            tensorboard_dir = f"logs/tensorboard/{label2name[label]}"
            if os.path.exists(tensorboard_dir):
                entropy_data[label] = load_entropy_data(tensorboard_dir, max_step)
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()  # Create second y-axis
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot accuracy data on left axis
    for i, (label, data) in enumerate(accuracy_data.items()):
        steps = data['steps']
        means = data['means']
        stds = data['stds']
        
        color = colors[i % len(colors)]
        
        # Plot accuracy line
        line1 = ax1.plot(steps, means, marker='o', linewidth=2, markersize=6,
                        label=f'{label} (Pass@1)', color=color, linestyle='-')
        
        # Fill between for accuracy
        means_array = np.array(means)
        stds_array = np.array(stds)
        ax1.fill_between(steps, means_array - stds_array, means_array + stds_array, 
                        alpha=0.2, color=color)
    
    # Plot entropy data on right axis
    for i, label in enumerate(show_entropy):
        if label in entropy_data and entropy_data[label]:
            color = colors[i % len(colors)]
            
            # Sort entropy data by step
            sorted_entropy = sorted(entropy_data[label].items())
            entropy_steps = [step for step, _ in sorted_entropy]
            entropy_values = [value for _, value in sorted_entropy]
            
            # Plot entropy line with dashed style
            line2 = ax2.plot(entropy_steps, entropy_values, marker='s', linewidth=2, 
                           markersize=4, label=f'{label} (Entropy)', color=color, 
                           linestyle='--', alpha=0.8)
    
    # Set labels and formatting
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pass@1(%)', color='black')
    ax2.set_ylabel('Entropy Loss', color='red')
    
    # Set colors for y-axis labels
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax1.grid(True, alpha=0.3)
    # plt.title('Training Curves with Entropy Loss')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_root, 'show_entropy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Remove temporary file
    temp_path = os.path.join(output_root, 'temp_entropy.png')
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"Entropy plot with dual axis saved to: {output_path}")
    
    return accuracy_data


def generate_show_minif2f_plot(log_fp, output_root, max_step=74):
    """Generate plot for show_minif2f comparison (minif2f data only)"""
    return generate_training_curves_plot(
        log_fp=log_fp,
        output_root=output_root,
        labels_to_show=show_minif2f,
        output_filename='show_minif2f',
        max_step=max_step,
        dataset_filter='minif2f_test'
    )

if __name__ == "__main__":
    fire.Fire(main)