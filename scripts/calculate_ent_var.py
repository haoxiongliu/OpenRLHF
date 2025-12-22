from tensorboard.backend.event_processing import event_accumulator
import glob
import os
import ast
import re
import numpy as np
import fire

# Label to model name mapping (from generate_paper_plot.py)
label2name = {
    "proofaug+": "1001-1-plpo-sum-rloo-mix6-max2depth-cons-cl0.2-0.28",
    "grpo-loo+direct": "0831-2-mix6-remove-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6"
}


def load_entropy_data(tensorboard_dir, max_step=80):
    """Load entropy loss data from tensorboard logs, merging data from multiple tfevents files"""
    entropy_data = {}
    
    # Find all event files in the directory
    event_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
    
    if not event_files:
        return entropy_data
    
    # Sort event files by modification time to process in chronological order
    event_files.sort(key=os.path.getmtime)
    
    # Process all event files and merge the data
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Check if entropy_loss exists
            if "train/entropy_loss" in ea.Tags()["scalars"]:
                scalars = ea.Scalars("train/entropy_loss")
                
                for scalar in scalars:
                    step = scalar.step
                    if step <= max_step:
                        # For overlapping steps, use the data from the later file (overwrite)
                        entropy_data[step] = scalar.value
                        
        except Exception as e:
            print(f"Error loading entropy data from {event_file}: {e}")
            continue
    
    return entropy_data


def load_evaluation_data(log_fp, label, max_step=80):
    """Load evaluation accuracy data from summary.log for a given label"""
    # Read and parse data (Python dict format, not JSON)
    data = []
    with open(log_fp, 'r') as f:
        for line in f:
            try:
                item = ast.literal_eval(line.strip())
                data.append(item)
            except (ValueError, SyntaxError):
                continue
    
    model_path_pattern = label2name[label]
    
    # Filter data for this label
    label_data = []
    for item in data:
        model_path = item.get('model', '')
        output_dir = item.get('output_dir', '')
        
        if model_path_pattern in model_path:
            # Extract global_step from model path
            step_match = re.search(r'global_step(\d+)', model_path)
            if step_match:
                step = int(step_match.group(1))
                if step > max_step:
                    continue
            else:
                continue
            
            # Filter for pset_test and -orig
            if 'pset_test' in output_dir and '-orig' in output_dir:
                # Extract seed from output_dir
                seed_match = None
                for part in output_dir.split('-'):
                    if part.startswith('s') and part[1:].isdigit():
                        seed_match = int(part[1:])
                        break
                
                if seed_match is not None:
                    accuracy = float(item.get('accuracy', 0))
                    passk = item['n']
                    
                    # Only use n=1 data
                    if passk == 1:
                        label_data.append({
                            'step': step,
                            'seed': seed_match,
                            'accuracy': accuracy,
                        })
    
    # Group by step and calculate mean accuracy across seeds
    step_groups = {}
    for item in label_data:
        step = item['step']
        if step not in step_groups:
            step_groups[step] = []
        step_groups[step].append(item['accuracy'])
    
    # Calculate mean accuracy for each step
    step_accuracies = {}
    for step in sorted(step_groups.keys()):
        accuracies = step_groups[step]
        step_accuracies[step] = np.mean(accuracies)
    
    return step_accuracies


def calculate_baseline(step_accuracies, step):
    """Calculate baseline for a given step as average of left, right, and self"""
    sorted_steps = sorted(step_accuracies.keys())
    
    if step not in sorted_steps:
        return None
    
    step_idx = sorted_steps.index(step)
    
    # Find left and right neighbors
    left_step = sorted_steps[step_idx - 1] if step_idx > 0 else None
    right_step = sorted_steps[step_idx + 1] if step_idx < len(sorted_steps) - 1 else None
    
    if left_step is None or right_step is None:
        return None
    
    # Baseline = average of (left, right, self)
    baseline = (step_accuracies[left_step] + step_accuracies[right_step] + step_accuracies[step]) / 3.0
    return baseline


def calculate_variance(step_accuracies, min_step=0, max_step=80):
    """Calculate cumulative variance estimate for steps min_step to max_step (excluding first and last)"""
    sorted_steps = sorted([s for s in step_accuracies.keys() if min_step <= s <= max_step])
    
    if len(sorted_steps) < 3:
        print(f"Warning: Not enough data points (need at least 3, got {len(sorted_steps)})")
        return None
    
    # Exclude first and last steps from processing
    # But they can still be used as neighbors for other steps
    first_step = sorted_steps[0]
    last_step = sorted_steps[-1]
    
    # Process steps that have both left and right neighbors
    # This means we skip the first and last steps themselves
    deviations = []
    for step in sorted_steps:
        # Skip first and last steps
        if step == first_step or step == last_step:
            continue
        
        baseline = calculate_baseline(step_accuracies, step)
        if baseline is not None:
            deviation = step_accuracies[step] - baseline
            deviations.append(deviation)
            print(f"  Step {step}: accuracy={step_accuracies[step]:.4f}, baseline={baseline:.4f}, deviation={deviation:.4f}")
    
    if len(deviations) == 0:
        return None
    
    # Calculate variance: mean of squared deviations
    cumulative_variance = np.mean(np.array(deviations) ** 2)
    return cumulative_variance


def main(log_fp="results/summary.log", max_step=80):
    """Calculate entropy at step 80 and cumulative variance for proofaug+ and grpo-loo+direct"""
    
    results = {}
    
    for label in ["proofaug+", "grpo-loo+direct"]:
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"{'='*60}")
        
        # Load entropy data
        tensorboard_dir = f"logs/tensorboard/{label2name[label]}"
        if not os.path.exists(tensorboard_dir):
            print(f"Warning: Tensorboard directory not found: {tensorboard_dir}")
            entropy_at_step = None
        else:
            entropy_data = load_entropy_data(tensorboard_dir, max_step=max_step)
            if max_step in entropy_data:
                entropy_at_step = entropy_data[max_step]
                print(f"Entropy at step {max_step}: {entropy_at_step:.6f}")
            else:
                # Find closest step
                available_steps = sorted([s for s in entropy_data.keys() if s <= max_step])
                if available_steps:
                    closest_step = available_steps[-1]
                    entropy_at_step = entropy_data[closest_step]
                    print(f"Entropy at step {max_step} not found. Using closest step {closest_step}: {entropy_at_step:.6f}")
                else:
                    entropy_at_step = None
                    print(f"Warning: No entropy data found for step <= {max_step}")
        
        # Load evaluation data
        step_accuracies = load_evaluation_data(log_fp, label, max_step=max_step)
        
        if not step_accuracies:
            print(f"Warning: No evaluation data found for {label}")
            variance = None
        else:
            print(f"Found {len(step_accuracies)} evaluation data points")
            print(f"Steps: {sorted(step_accuracies.keys())}")
            
            # Calculate variance
            variance = calculate_variance(step_accuracies, min_step=0, max_step=max_step)
            if variance is not None:
                print(f"Cumulative variance (steps 20 to {max_step}): {variance:.6f}")
            else:
                print(f"Warning: Could not calculate variance")
        
        results[label] = {
            'entropy_at_step_80': entropy_at_step,
            'cumulative_variance': variance,
            'step_accuracies': step_accuracies
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, result in results.items():
        print(f"\n{label}:")
        if result['entropy_at_step_80'] is not None:
            print(f"  Entropy at step {max_step}: {result['entropy_at_step_80']:.6f}")
        else:
            print(f"  Entropy at step {max_step}: N/A")
        
        if result['cumulative_variance'] is not None:
            print(f"  Cumulative variance: {result['cumulative_variance']:.6f}")
        else:
            print(f"  Cumulative variance: N/A")
    
    return results


if __name__ == "__main__":
    fire.Fire(main)

