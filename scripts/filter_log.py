#!/usr/bin/env python3
"""
Script to filter out lines containing "0807-q2515bi" from summary.log
and save the filtered content to a new log file.
Also saves the excluded lines to a separate log file.
"""

import os
import sys
from datetime import datetime
import fire

def filter_log_file(pattern, input_file="results/summary.log"):
    """
    Filter lines containing the specified pattern from input file.
    
    Args:
        input_file (str): Path to input log file
        output_filtered (str): Path to output file for lines WITHOUT the pattern
        output_excluded (str): Path to output file for lines WITH the pattern
        pattern (str):g Pattern to filter out
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output file paths
    output_filtered = f"results/summary_filtered_{timestamp}.log"
    output_excluded = f"results/summary_excluded_{timestamp}.log"
   

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        filtered_lines = []
        excluded_lines = []
        
        # Separate lines based on whether they contain the pattern
        for line in lines:
            if pattern in line:
                excluded_lines.append(line)
            else:
                filtered_lines.append(line)
        
        # Write filtered lines (without pattern) to output file
        with open(output_filtered, 'w', encoding='utf-8') as outfile:
            outfile.writelines(filtered_lines)
        
        # Write excluded lines (with pattern) to separate file
        with open(output_excluded, 'w', encoding='utf-8') as outfile:
            outfile.writelines(excluded_lines)
        
        # Print summary
        total_lines = len(lines)
        filtered_count = len(filtered_lines)
        excluded_count = len(excluded_lines)
        
        print(f"Filtering completed successfully!")
        print(f"Total lines in original file: {total_lines}")
        print(f"Lines without '{pattern}': {filtered_count}")
        print(f"Lines with '{pattern}' (excluded): {excluded_count}")
        print(f"Filtered content saved to: {output_filtered}")
        print(f"Excluded content saved to: {output_excluded}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    fire.Fire(filter_log_file)