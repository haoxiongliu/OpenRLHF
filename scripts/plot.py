#!/usr/bin/env python3
"""
Training Curves Analysis for summary.log with Error Bars
统计summary.log里的，对同一训练同义hammer相同的n，绘制按步数的曲线图
按model分组，同model同n同hammer画在一条曲线上
只画有step的模型，并将hammer列表转换为recipe名字
使用-s\d+模式识别种子差异，计算标准差并添加误差条
"""

import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import ast
from os.path import basename

def remove_seed_from_params(inference_params):
    """Remove seed information (-s\d+) from inference parameters"""
    if not inference_params:
        return inference_params
    
    # Remove seed pattern -s followed by digits
    cleaned_params = re.sub(r'-s\d+', '', inference_params)
    
    # Clean up any double slashes or leading/trailing slashes that might be left
    cleaned_params = re.sub(r'/+', '/', cleaned_params)
    cleaned_params = cleaned_params.strip('/')
    
    return cleaned_params

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

def aggregate_by_step(points_list):
    """Aggregate points by step, calculating mean and std across seeds"""
    step_data = defaultdict(list)
    
    # Group by step
    for point in points_list:
        if not point.get('isolated', False):  # Skip isolated baseline points
            step_data[point['step']].append(point['pass_rate'])
    
    # Calculate mean and std for each step
    aggregated = []
    for step, pass_rates in step_data.items():
        if len(pass_rates) > 0:
            mean_pass_rate = np.mean(pass_rates)
            std_pass_rate = np.std(pass_rates) if len(pass_rates) > 1 else 0
            aggregated.append({
                'step': step,
                'mean_pass_rate': mean_pass_rate,
                'std_pass_rate': std_pass_rate,
                'num_seeds': len(pass_rates)
            })
    
    return sorted(aggregated, key=lambda x: x['step'])

def main():
    # Parse the log file
    print("Parsing summary.log...")
    data = parse_log_file('results/summary.log')
    print(f"Found {len(data)} entries")
    
    # Group data by model, n, and recipe (with seed-cleaned params)
    grouped_data = defaultdict(list)
    
    # Collect baseline data (step 0) from kimina base model
    baseline_data = {}
    
    for entry in data:
        model_path = entry.get('model', '')
        output_dir = entry.get('output_dir', '')
        step = extract_step_number(model_path)
        inference_params = extract_inference_params(output_dir)
        
        # Ad-hoc static baseline data
        if 'hf_models' in model_path and step is None:
            n = entry.get('n', 'unknown')
            
            # Clean seed from baseline params too
            real_output_dir = remove_seed_from_params(output_dir)
            
            # Use output_dir directly for baseline distinction if inference_params is empty
            if not real_output_dir:
                # If inference_params extraction failed, use the output_dir directly
                baseline_key = (n, output_dir)
            else:
                baseline_key = (n, real_output_dir)
            
            # Store multiple baseline values for the same key (different seeds)
            if baseline_key not in baseline_data:
                baseline_data[baseline_key] = []
            
            baseline_data[baseline_key].append({
                'step': 0,
                'model_path': model_path,
                'output_dir': output_dir,
                'accuracy': entry.get('correct', 0),
                'total': entry.get('total', 0),
                'pass_rate': entry.get('correct', 0) / max(entry.get('total', 1), 1)
            })
        
        # Only include entries with step numbers for training curves
        if step is None:
            continue
            
        base_model = extract_model_info(model_path)
        training_name = extract_training_name(model_path)
        n = entry.get('n', 'unknown')
        
        # Clean seed from inference params for grouping
        cleaned_params = remove_seed_from_params(inference_params)
        
        # Debug output
        if inference_params != cleaned_params:
            print(f"DEBUG Training - Original: '{inference_params}' -> Cleaned: '{cleaned_params}'")
        
        # Create a key for grouping (only training_name, n, and cleaned inference_params)
        key = (training_name, n, cleaned_params)
        
        grouped_data[key].append({
            'step': step,
            'model_path': model_path,
            'accuracy': entry.get('correct', 0),  # 使用'correct'字段作为accuracy
            'total': entry.get('total', 0),
            'pass_rate': entry.get('correct', 0) / max(entry.get('total', 1), 1)
        })
    
    # Process baseline data - calculate mean and std for each baseline key
    processed_baseline_data = {}
    for baseline_key, baseline_points in baseline_data.items():
        if len(baseline_points) > 0:
            pass_rates = [p['pass_rate'] for p in baseline_points]
            processed_baseline_data[baseline_key] = {
                'mean_pass_rate': np.mean(pass_rates),
                'std_pass_rate': np.std(pass_rates) if len(pass_rates) > 1 else 0,
                'num_seeds': len(pass_rates),
                'step': 0
            }
    
    # Aggregate grouped data by step (across seeds)
    aggregated_data = {}
    for key, points in grouped_data.items():
        aggregated_data[key] = aggregate_by_step(points)
    
    # Debug output - show all grouping keys
    print(f"\nDEBUG - Baseline grouping keys:")
    for baseline_key, baseline_points in baseline_data.items():
        print(f"  {baseline_key}: {len(baseline_points)} points")
    
    print(f"\nDEBUG - Training grouping keys:")
    for key, points in grouped_data.items():
        print(f"  {key}: {len(points)} points")
    
    print(f"Found {len(processed_baseline_data)} baseline groups")
    for baseline_key, data_point in processed_baseline_data.items():
        print(f"  Baseline: {baseline_key} -> {data_point['mean_pass_rate']:.3f}±{data_point['std_pass_rate']:.3f} pass rate ({data_point['num_seeds']} seeds)")
    
    if not aggregated_data:
        print("No training curves with steps found!")
        return
    
    # Print debug information
    print("\nTraining curves found:")
    for key, points in aggregated_data.items():
        training_name, n, inference_params = key
        steps = [p['step'] for p in points]
        num_seeds = [p['num_seeds'] for p in points]
        print(f"  {training_name} n={n}{inference_params}: {len(points)} points, steps={steps}, seeds={num_seeds}")
    
    # Get unique n values
    n_values = set()
    for key in aggregated_data.keys():
        training_name, n, inference_params = key
        n_values.add(n)
    for baseline_key in processed_baseline_data.keys():
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
    for key in aggregated_data.keys():
        training_name, n, inference_params = key
        training_names.add(training_name)
    
    # Create color map for training names
    training_colors = plt.cm.tab10(range(len(training_names)))
    color_map = dict(zip(training_names, training_colors))
    
    # Create color map for baseline recipes
    baseline_recipes = set()
    for baseline_key in processed_baseline_data.keys():
        n, inference_params_or_dir = baseline_key
        baseline_recipes.add(inference_params_or_dir)
    baseline_colors = plt.cm.Set2(range(len(baseline_recipes)))
    baseline_color_map = dict(zip(baseline_recipes, baseline_colors))
    
    # Plot for each n value
    for i, n_val in enumerate(n_values):
        ax = axes[i]
        
        # Plot baseline points for this n value
        for baseline_key, baseline_point in processed_baseline_data.items():
            n, inference_params_or_dir = baseline_key
            if n != n_val:
                continue
                
            mean_pass_rate = baseline_point['mean_pass_rate']
            std_pass_rate = baseline_point['std_pass_rate']
            num_seeds = baseline_point['num_seeds']
            
            # Simplify baseline label - only keep the last part after the last '/'
            if inference_params_or_dir and '/' in inference_params_or_dir:
                baseline_label = inference_params_or_dir.split('/')[-1]
            else:
                baseline_label = inference_params_or_dir or 'default'
            
            # Get color for this baseline recipe
            baseline_color = baseline_color_map[inference_params_or_dir]
            
            # Plot baseline at step=0 with shaded area for error
            ax.scatter([0], [mean_pass_rate], marker='x', s=64, color=baseline_color,
                      label=f'{baseline_label} baseline ({num_seeds} seeds)', zorder=5)
            
            # Add shaded area for baseline standard deviation (very narrow, almost like a vertical line)
            if std_pass_rate > 0:
                step_width = 0.5  # Very narrow horizontal extent, almost like a vertical line
                ax.fill_between([-step_width, step_width], 
                               [mean_pass_rate - std_pass_rate] * 2,
                               [mean_pass_rate + std_pass_rate] * 2, 
                               alpha=0.2, color=baseline_color)
        
        # Plot training curves for this n value
        for key, points in aggregated_data.items():
            training_name, n, inference_params = key
            if n != n_val:
                continue
                
            if not points:
                continue
                
            # Include training name and inference_params in label (n is already in title)
            label = f'{training_name}{inference_params}'
            
            # Get color for this training
            color = color_map[training_name]
            
            # Extract data for plotting
            steps = [p['step'] for p in points]
            mean_pass_rates = [p['mean_pass_rate'] for p in points]
            std_pass_rates = [p['std_pass_rate'] for p in points]
            num_seeds = [p['num_seeds'] for p in points]
            
            # Add number of seeds info to label
            max_seeds = max(num_seeds) if num_seeds else 1
            min_seeds = min(num_seeds) if num_seeds else 1
            if max_seeds == min_seeds:
                seed_info = f" ({max_seeds} seeds)"
            else:
                seed_info = f" ({min_seeds}-{max_seeds} seeds)"
            
            # Plot with shaded area for error bars
            if len(steps) > 1:  # Only plot line if we have multiple points
                # Plot main line
                ax.plot(steps, mean_pass_rates, 'o-', label=label + seed_info, 
                       linewidth=2, markersize=6, color=color)
                # Plot shaded area for standard deviation
                upper_bound = [mean + std for mean, std in zip(mean_pass_rates, std_pass_rates)]
                lower_bound = [mean - std for mean, std in zip(mean_pass_rates, std_pass_rates)]
                ax.fill_between(steps, lower_bound, upper_bound, alpha=0.2, color=color)
            elif len(steps) == 1:
                # For single point, still use error bar
                ax.errorbar(steps, mean_pass_rates, yerr=std_pass_rates,
                           fmt='o', label=label + seed_info, markersize=6, 
                           color=color, capsize=3)
        
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