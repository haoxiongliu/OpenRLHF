#!/usr/bin/env python3
"""
Training Curves Analysis for summary.log
统计summary.log里的，对同一训练同义hammer相同的n，绘制按步数的曲线图
按model分组，同model同n同hammer画在一条曲线上
只画有step的模型，并将hammer列表转换为recipe名字
"""

import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import ast

# No longer need RECIPE2HAMMER_LIST as we use hammer_recipe directly

# Remove hammers_to_recipe function - no longer needed as we use hammer_recipe directly

def parse_log_file(filename):
    """Parse the summary.log file and return a list of entries"""
    data = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Use ast.literal_eval for safer parsing of Python dict format
                entry = ast.literal_eval(line.strip())
                data.append(entry)
            except Exception as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    return data

def extract_step_number(model_path):
    """Extract step number from model path"""
    # Look for global_step pattern
    match = re.search(r'global_step(\d+)_hf', model_path)
    if match:
        return int(match.group(1))
    
    return None

def extract_model_info(model_path):
    """Extract model information from model path"""
    # Remove step information to get base model
    base_model = re.sub(r'/global_step\d+_hf/?', '', model_path)
    return base_model.strip('/')

def extract_training_name(model_path):
    """Extract training name from model path using date pattern"""
    # For checkpoint models, extract the training run name by date pattern
    if 'checkpoints/ckpts/' in model_path:
        parts = model_path.split('/')
        for part in parts:
            # Look for date pattern (4 digits for month/day)
            if re.match(r'\d{4}-', part):
                return part
    return "unknown"

def extract_inference_params(output_dir):
    """Extract inference parameters from output_dir after global_step_xxx_hf"""
    # Look for pattern global_step[number]_hf followed by inference params
    match = re.search(r'global_step\d+_hf(.+)', output_dir)
    if match:
        return match.group(1)
    return None

def main():
    # Parse the log file
    print("Parsing summary.log...")
    data = parse_log_file('results/summary.log')
    print(f"Found {len(data)} entries")
    
    # Group data by model, n, and recipe
    grouped_data = defaultdict(list)
    
    # Collect baseline data (step 0) from kimina base model
    baseline_data = {}
    
    for entry in data:
        model_path = entry.get('model', '')
        output_dir = entry.get('output_dir', '')
        step = extract_step_number(model_path)
        inference_params = extract_inference_params(output_dir)
        
        # Collect baseline data from kimina base model
        if 'Kimina-Prover-Preview-Distill-1.5B' in model_path and step is None:
            n = entry.get('n', 'unknown')
            
            # Use output_dir directly for baseline distinction if inference_params is empty
            if not inference_params:
                # If inference_params extraction failed, use the output_dir directly
                baseline_key = (n, output_dir)
            else:
                baseline_key = (n, inference_params)
                
            baseline_data[baseline_key] = {
                'step': 0,
                'model_path': model_path,
                'output_dir': output_dir,
                'accuracy': entry.get('correct', 0),
                'total': entry.get('total', 0),
                'pass_rate': entry.get('correct', 0) / max(entry.get('total', 1), 1)
            }
        
        # Only include entries with step numbers for training curves
        if step is None:
            continue
            
        base_model = extract_model_info(model_path)
        training_name = extract_training_name(model_path)
        n = entry.get('n', 'unknown')
        
        # Create a key for grouping (only training_name, n, and inference_params)
        key = (training_name, n, inference_params)
        
        grouped_data[key].append({
            'step': step,
            'model_path': model_path,
            'accuracy': entry.get('correct', 0),  # 使用'correct'字段作为accuracy
            'total': entry.get('total', 0),
            'pass_rate': entry.get('correct', 0) / max(entry.get('total', 1), 1)
        })
    
    # Add baseline data (step 0) as isolated points for training curves
    for key in list(grouped_data.keys()):
        training_name, n, inference_params = key
        
        # Try both inference_params and potential output_dir matches
        baseline_key = (n, inference_params)
        if baseline_key not in baseline_data:
            # If exact match not found, look for baseline with same n and any inference_params
            for baseline_n, baseline_params in baseline_data.keys():
                if baseline_n == n:
                    # Found a baseline with same n, use it even if inference_params don't exactly match
                    baseline_key = (baseline_n, baseline_params)
                    break
        
        if baseline_key in baseline_data:
            # Add baseline as isolated point (not connected to training curve)
            baseline_point = baseline_data[baseline_key].copy()
            baseline_point['isolated'] = True
            grouped_data[key].append(baseline_point)
    
    print(f"Found {len(baseline_data)} baseline data points for step 0")
    for baseline_key, data_point in baseline_data.items():
        print(f"  Baseline: {baseline_key} -> {data_point['pass_rate']:.3f} pass rate")
    
    if not grouped_data:
        print("No training curves with steps found!")
        return
    
    # Sort data points by step for each group
    for key in grouped_data:
        grouped_data[key].sort(key=lambda x: x['step'])
    
    # Print debug information
    print("\nTraining curves found:")
    for key, points in grouped_data.items():
        training_name, n, inference_params = key
        steps = [p['step'] for p in points]
        print(f"  {training_name} n={n}{inference_params}: {len(points)} points, steps={steps}")
    
    print(f"\nBaseline data (step 0):")
    for baseline_key, data_point in baseline_data.items():
        n, inference_params = baseline_key
        print(f"  n={n}{inference_params}: {data_point['pass_rate']:.3f} pass rate, {data_point['accuracy']} solved")
    
    # Get unique n values
    n_values = set()
    for key in grouped_data.keys():
        training_name, n, inference_params = key
        n_values.add(n)
    for baseline_key in baseline_data.keys():
        n, inference_params_or_dir = baseline_key
        n_values.add(n)
    
    n_values = sorted(list(n_values))
    print(f"Found n values: {n_values}")
    
    # Create subplots for each n value
    n_count = len(n_values)
    if n_count == 0:
        print("No n values found!")
        return
    
    fig, axes = plt.subplots(1, n_count, figsize=(12 * n_count, 10))
    if n_count == 1:
        axes = [axes]  # Make it a list for consistency
    
    # Get all unique training names and assign colors
    training_names = set()
    for key in grouped_data.keys():
        training_name, n, inference_params = key
        training_names.add(training_name)
    
    # Create color map for training names
    training_colors = plt.cm.tab10(range(len(training_names)))
    color_map = dict(zip(training_names, training_colors))
    
    # Plot for each n value
    for i, n_val in enumerate(n_values):
        ax = axes[i]
        
        # Plot baseline points for this n value
        for baseline_key, baseline_point in baseline_data.items():
            n, inference_params_or_dir = baseline_key
            if n != n_val:
                continue
                
            pass_rate = baseline_point['pass_rate']
            
            # Simplify baseline label - only keep the last part after the last '/'
            if inference_params_or_dir and '/' in inference_params_or_dir:
                baseline_label = inference_params_or_dir.split('/')[-1]
            else:
                baseline_label = inference_params_or_dir or 'default'
            
            # Plot baseline at step=0
            ax.scatter([0], [pass_rate], s=80, marker='x', color='gray',
                      label=f'{baseline_label} baseline')
        
        # Plot training curves for this n value
        for key, points in grouped_data.items():
            training_name, n, inference_params = key
            if n != n_val:
                continue
                
            # Only plot connected points (training curve), skip isolated baselines since we plotted them above
            connected_points = [p for p in points if not p.get('isolated', False)]
            
            if not connected_points:
                continue
                
            # Include training name and inference_params in label (n is already in title)
            label = f'{training_name}{inference_params}'
            
            # Get color for this training
            color = color_map[training_name]
            
            # Plot connected points (training curve)
            steps = [p['step'] for p in connected_points]
            pass_rates = [p['pass_rate'] for p in connected_points]
            
            if len(steps) > 1:  # Only plot line if we have multiple points
                ax.plot(steps, pass_rates, 'o-', label=label, linewidth=2, markersize=6, color=color)
            elif len(steps) == 1:
                ax.scatter(steps, pass_rates, label=label, s=50, color=color)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Pass Rate')
        ax.set_title(f'n={n_val}')
        ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Leave space for legend on the top
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to results/training_curves.png")
    plt.show()

if __name__ == "__main__":
    main()