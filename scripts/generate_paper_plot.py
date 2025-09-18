from matplotlib import pyplot as plt
import json
import os
import fire
import numpy as np
import ast
import re
from tensorboard.backend.event_processing import event_accumulator
import glob
# "ppo_0.2_0.28": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6",
label2name = {
    # "gspo": "0901-3-record_pa_reward-mix6-gspo-n8-rloo-3072-kl0.0-cl0.2-0.27-trT0.6",
    # "gspo_0.1_0.12": "0902-1-record_pa_reward-mix6-gspo-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "ppo": "0903-1-record_pa_reward-mix6-ppo-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.2_0.28": "0903-1-record_pa_reward-mix6-ppo-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.2_0.28+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",  # best one seems
    "ppo_0.2_0.28+aggr": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo_0.5_1.0": "0819-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.5-1.0-trT0.6",
    "ppo_0.5_1.0+aggr": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-mix6-max2depth-0807-3072-kl0.0-cl0.5-1.0-trT0.6",  
    "ppo+direct": "0831-2-mix6-remove-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6", 
    "ppo_0.2_0.28+direct": "0831-2-mix6-remove-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6", # preliminary
    "plmo_single_0.2_0.28": "0831-3-record_pa_reward-mix6-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_single_0.2_0.28+cons": "0826-1-mix6-max2depth-cons-0821-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6", # nearest one
    "plmo_single+aggr": "0831-1-mix6-max2depth-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo w/o rc": "0830-1-record_pa_reward-mix6-plmo-average-n8-rloo-3072-kl0.0-cl1.0-1e09-trT0.6",
    "plmo_avg": "0901-1-record_pa_reward-mix6-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_avg_0.1_0.12": "0901-1-record_pa_reward-mix6-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_avg_0.1_0.12+cons": "0828-2-mix6-max2depth-cons-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_sum": "0828-1-record_pa_reward-mix6-plmo-sum-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_sum_0.2_0.28": "0828-1-record_pa_reward-mix6-plmo-sum-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_sum_0.2_0.28+cons": "0827-1-mix6-max2depth-cons-plmo-sum-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6",
    "plmo_avg+pab+cons": "0901-2-mix6-max2depth-cons-pab-plmo-average-n8-rloo-3072-kl0.0-cl0.1-0.12-trT0.6",
    "plmo_sum_0.3_0.4": "0905-1-plmo-sum-record_pa_reward-mix6-cl0.3-0.4",
    "plmo_sum_0.4_0.6+cons": "0905-2-plmo-sum-mix6-max2depth-cons-cl0.4-0.6",
    "plmo_avg_0.05_0.06+cons": "0907-1-plmo-average-mix6-max2depth-cons-cl0.05-0.06",
    "plmo_avg_0.03_0.03+cons": "0907-2-plmo-average-mix6-max2depth-cons-cl0.03-0.03",
    "osppo w/o rc": "0908-1-osppo-average-record_pa-mix6-cl0.2-0.27",
    "osppo_sum_0.2_0.28": "0909-1-osppo-sum-record_pa-mix6-cl0.2-0.28",
    "osppo_sum_0.2_0.28+cons": "0910-1-osppo-sum-mix6-max2depth-cons-cl0.2-0.28",
    "osppo_sum_pab_0.2_0.28+cons": "0914-1-osppo-sum-pab-mix6-max2depth-cons-cl0.2-0.28",
    "osppo_sum_pab_0.2_0.28+cons+nt": "0915-1-osppo-sum-pab-mix7-max2depth-cons-nt-cl0.2-0.28",
    "grpo": "0917-1-ppo-group_norm-record_pa-mix6-kl0.04-cl0.2-0.2",
    "gspo": "0917-2-gspo-group_norm-record_pa-mix6-cl0.2-0.27",
    "gspo_sum_0.2_0.28-nt": "0917-3-gspo-sum-mix7-max2depth-cons-nt-cl0.2-0.28",
    "gspo_sum_0.2_0.28": "0918-1-gspo-sum-record_pa-mix6-cl0.2-0.28",

}

def main(log_fp="results/summary.log", output_root="results/paper_plot", paper=False):
    """
    Generate the ProofAug+ plot from the paper
    and save the pngs into output_root
    """

    # ppo_cons, plmo_single+cons, plmo_avg+cons ppo+aggr+large
    # "gspo", "gspo_0.1_0.12", 
    os.makedirs(output_root, exist_ok=True)
    if not paper:
        preliminary = ["ppo", "ppo+direct", "gspo", "osppo w/o rc", "gspo_sum_0.2_0.28"]
        show_plmo = ["ppo_0.2_0.28", "plmo w/o rc", "plmo_single_0.2_0.28", "plmo_avg_0.1_0.12", "plmo_sum_0.2_0.28", "osppo w/o rc", "osppo_sum_0.2_0.28", "grpo", "gspo", "gspo_sum_0.2_0.28"] # "ppo_wo_rc",
        show_aggr_cons = ["ppo_0.2_0.28", "ppo_0.2_0.28+direct", "ppo_0.2_0.28+aggr", "ppo_0.5_1.0+aggr", "ppo_0.2_0.28+cons"] # "ppo_0.5_1.0",
        show_plmo_cons = ["ppo_0.2_0.28", "ppo_0.2_0.28+direct", "ppo_0.2_0.28+cons", "plmo_sum_0.2_0.28+cons", "plmo_avg_0.1_0.12+cons", "plmo_single_0.2_0.28+cons", "osppo_sum_0.2_0.28", "osppo_sum_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons+nt", "gspo_sum_0.2_0.28-nt"] 
        show_minif2f = ["ppo", "ppo+cons", "plmo_single_0.2_0.28+cons", "plmo_avg_0.1_0.12+cons", "osppo_sum_0.2_0.28", "osppo_sum_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons+nt"] # limitation part
        show_entropy = ["ppo+cons", "ppo", "osppo_sum_0.2_0.28", "osppo_sum_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons", "grpo", "gspo", "gspo_sum_0.2_0.28-nt"] # , "osppo_sum_pab_0.2_0.28+cons+nt"
        # "plmo_avg_0.1_0.12+cons", "plmo_sum_0.2_0.28+cons", 这些再做个ablation 就行 
        # "plmo_single_0.2_0.28+cons", "plmo_sum_0.4_0.6+cons", , "plmo_avg_0.05_0.06+cons" seed  原因逃过一劫？ "osppo w/o rc", "plmo_avg_0.03_0.03+cons", 
    # Generate show_plmo plot

        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=preliminary,
            output_filename='preliminary',
            max_step=48
        )

        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=show_plmo,
            output_filename='show_plmo',
            max_step=400
        )

        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=show_plmo_cons,
            output_filename='show_plmo_cons',
            max_step=400
        )

        # Generate show_aggr_cons plot
        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=show_aggr_cons,
            output_filename='show_aggr_cons',
            max_step=100
        )
        
        # Generate show_entropy plot
        generate_show_entropy_plot(
            log_fp, output_root, show_entropy, 'show_entropy',
            max_step=400
        )
        
        # Generate show_minif2f plot
        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=show_minif2f,
            output_filename='show_minif2f',
            max_step=400,
            dataset_filter='minif2f_test'
        )
    else:
        preliminary = ["ppo", "ppo+direct"]
        show_plmo = ["ppo_0.2_0.28",  "osppo w/o rc", "osppo_sum_0.2_0.28"] # "ppo_wo_rc", # "plmo w/o rc", "plmo_single_0.2_0.28", "plmo_avg_0.1_0.12", "plmo_sum_0.2_0.28",
        show_aggr_cons = [] # give up
        show_plmo_cons = ["ppo_0.2_0.28", "ppo_0.2_0.28+direct", "ppo_0.2_0.28+cons", "plmo_sum_0.2_0.28+cons", "plmo_avg_0.1_0.12+cons", "plmo_single_0.2_0.28+cons", "osppo_sum_0.2_0.28", "osppo_sum_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons+nt"] 
        show_minif2f = ["ppo", "ppo+cons", "plmo_single_0.2_0.28+cons", "plmo_avg_0.1_0.12+cons", "osppo_sum_0.2_0.28", "osppo_sum_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons+nt"] # limitation part
        show_entropy = ["ppo+cons", "ppo", "osppo_sum_0.2_0.28", "osppo_sum_0.2_0.28+cons"] 
        # , "osppo_sum_pab_0.2_0.28+cons", "osppo_sum_pab_0.2_0.28+cons+nt"        

        # generate_training_curves_plot(
        #     log_fp=log_fp,
        #     output_root=output_root,
        #     labels_to_show=preliminary,
        #     output_filename='preliminary',
        #     max_step=50
        # )
        generate_show_entropy_plot(
            log_fp, output_root, preliminary, 'preliminary',
            max_step=50
        )
        # generate_training_curves_plot(
        #     log_fp=log_fp,
        #     output_root=output_root,
        #     labels_to_show=show_plmo,
        #     output_filename='show_plmo',
        #     max_step=300
        # )
        generate_show_entropy_plot(
            log_fp, output_root, show_plmo, 'show_plmo',
            max_step=300
        )
        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=show_plmo_cons,
            output_filename='show_plmo_cons',
            max_step=400
        )

        # Generate show_aggr_cons plot
        # generate_training_curves_plot(
        #     log_fp=log_fp,
        #     output_root=output_root,
        #     labels_to_show=show_aggr_cons,
        #     output_filename='show_aggr_cons',
        #     max_step=100
        # )
        
        # Generate show_entropy plot
        generate_show_entropy_plot(
            log_fp, output_root, show_entropy, 'show_entropy',
            max_step=400
        )
        
        # Generate show_minif2f plot
        generate_training_curves_plot(
            log_fp=log_fp,
            output_root=output_root,
            labels_to_show=show_minif2f,
            output_filename='show_minif2f',
            max_step=400,
            dataset_filter='minif2f_test'
        )

    
    



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
        
        # Find matching entries for both T0.6 and T0.1
        label_data_t06 = []
        label_data_t01 = []
        
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
                    data_entry = {
                        'step': step,
                        'seed': seed_match,
                        'accuracy': accuracy,
                        'model': model_path,
                        'output_dir': output_dir
                    }
                    
                    # Check temperature parameter in output_dir
                    if '-T0.1-' in output_dir:
                        pass # no longer needed
                        # label_data_t01.append(data_entry)
                    elif '-T0.6-' in output_dir:
                        label_data_t06.append(data_entry)
                    else:
                        # Default to T0.6 if no temperature specified
                        label_data_t06.append(data_entry)
        
        # Add T0.6 data with original label
        if label_data_t06:
            results[label] = label_data_t06
            
        # Add T0.1 data with modified label
        if label_data_t01:
            results[f"{label}+T0.1"] = label_data_t01
    
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
    
    plt.xlabel('Iteration')
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




def load_entropy_data(tensorboard_dir, max_step=72):
    """Load entropy loss data from tensorboard logs, merging data from multiple tfevents files"""
    entropy_data = {}
    
    # Find all event files in the directory
    event_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
    
    if not event_files:
        return entropy_data
    
    # Sort event files by modification time to process in chronological order
    event_files.sort(key=os.path.getmtime)
    
    # Process all event files and merge the data
    processed_files = 0
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Check if entropy_loss exists
            if "train/entropy_loss" in ea.Tags()["scalars"]:
                scalars = ea.Scalars("train/entropy_loss")
                file_data_count = 0
                
                for scalar in scalars:
                    step = scalar.step
                    if step <= max_step:
                        # For overlapping steps, use the data from the later file (overwrite)
                        entropy_data[step] = scalar.value
                        file_data_count += 1
                
                if file_data_count > 0:
                    processed_files += 1
                        
        except Exception as e:
            print(f"Error loading entropy data from {event_file}: {e}")
            continue  # Continue processing other files even if one fails
    
    if processed_files > 1:
        print(f"Merged entropy data from {processed_files} tensorboard files in {tensorboard_dir}")
    
    return entropy_data


def generate_show_entropy_plot(log_fp, output_root, show_entropy, output_filename,max_step=72):
    """Generate plot for show_entropy comparison with dual y-axis for entropy, using continuous lines from tensorboard data"""
    
    # Get accuracy data using existing function
    accuracy_data = generate_training_curves_plot(
        log_fp=log_fp,
        output_root=output_root,
        labels_to_show=show_entropy,
        output_filename='temp_entropy',  # temporary file
        max_step=max_step
    )
    
    # Load entropy data for each label (including T0.1 variants from accuracy_data)
    entropy_data = {}
    for label in accuracy_data.keys():  # Use actual labels from accuracy_data instead of show_entropy
        # Remove the +T0.1 suffix to get the base label for tensorboard directory
        base_label = label.replace('+T0.1', '') if '+T0.1' in label else label
        if base_label in label2name:
            tensorboard_dir = f"logs/tensorboard/{label2name[base_label]}"
            if os.path.exists(tensorboard_dir):
                entropy_data[label] = load_entropy_data(tensorboard_dir, max_step)
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()  # Create second y-axis
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot accuracy data on left axis (keep as points with error bars)
    for i, (label, data) in enumerate(accuracy_data.items()):
        steps = data['steps']
        means = data['means']
        stds = data['stds']
        
        color = colors[i % len(colors)]
        
        # Plot accuracy line
        line1 = ax1.plot(steps, means, marker='o', linewidth=2, markersize=6,
                        label=label, color=color, linestyle='-')
        
        # Fill between for accuracy
        means_array = np.array(means)
        stds_array = np.array(stds)
        ax1.fill_between(steps, means_array - stds_array, means_array + stds_array, 
                        alpha=0.2, color=color)
    
    # Plot entropy data on right axis as continuous lines without markers
    for i, label in enumerate(accuracy_data.keys()):
        if label in entropy_data and entropy_data[label]:
            color = colors[i % len(colors)]
            
            # Sort entropy data by step for continuous line
            sorted_entropy = sorted(entropy_data[label].items())
            entropy_steps = [step for step, _ in sorted_entropy]
            entropy_values = [value for _, value in sorted_entropy]
            
            # Plot entropy as continuous line without markers, easier to distinguish
            line2 = ax2.plot(entropy_steps, entropy_values, linewidth=2, 
                           color=color, linestyle='-', alpha=0.8)
    
    # Set labels and formatting
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Pass@1(%)', color='black')
    ax2.set_ylabel('Entropy Loss', color='red')
    
    # Set colors for y-axis labels
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Use only accuracy lines for legend (entropy lines have no labels)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    
    ax1.grid(True, alpha=0.3)
    # plt.title('Training Curves with Entropy Loss')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_root, f'{output_filename}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Remove temporary file
    temp_path = os.path.join(output_root, 'temp_entropy.png')
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"Entropy plot with dual axis saved to: {output_path}")
    
    return accuracy_data



if __name__ == "__main__":
    fire.Fire(main)