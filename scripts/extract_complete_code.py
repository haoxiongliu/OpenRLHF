#!/usr/bin/env python3
"""
Script to extract complete code from code_compilation.json files.
Creates a folder in logs with a name indicating the source location,
then extracts only items where complete=true and saves each code to a file named with the 'name' field.
"""

import json
import os
import fire
from pathlib import Path


def extract_complete_code(json_file_path, logs_dir="logs"):
    """
    Extract complete code from a code_compilation.json file.
    
    Args:
        json_file_path: Path to the code_compilation.json file
        logs_dir: Directory where to create the output folder
    """
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} does not exist")
        return
    
    # Read the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Create folder name from the path
    path_parts = Path(json_file_path).parts
    # Extract meaningful parts from the path for folder naming
    folder_name_parts = []
    
    # Find the index where meaningful directory structure starts
    for i, part in enumerate(path_parts):
        if part == "results":
            # Take the next parts that are meaningful
            if i + 1 < len(path_parts):
                folder_name_parts.append(path_parts[i + 1])  # e.g., "minif2f_test_kimina"
            if i + 3 < len(path_parts):
                folder_name_parts.append(path_parts[i + 3])  # e.g., model name
            break
    
    if not folder_name_parts:
        # Fallback: use the parent directory name
        folder_name_parts = [Path(json_file_path).parent.name]
    
    folder_name = "_".join(folder_name_parts)
    
    # Create output directory
    output_dir = Path(logs_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    
    # Filter and extract complete code
    complete_count = 0
    total_count = 0
    
    for item in data:
        total_count += 1
        
        # Check if compilation_result exists and complete is true
        if "compilation_result" in item and item["compilation_result"].get("complete", False):
            complete_count += 1
            
            # Get the name and code
            name = item.get("name", f"unnamed_{total_count}")
            code = item.get("code", "")
            
            # Save to file
            file_path = output_dir / f"{name}.lean"
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                print(f"Saved: {file_path}")
            except Exception as e:
                print(f"Error saving {file_path}: {e}")
    
    print(f"\nSummary:")
    print(f"Total items: {total_count}")
    print(f"Complete items: {complete_count}")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(extract_complete_code)
