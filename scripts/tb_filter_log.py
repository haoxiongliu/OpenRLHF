#!/usr/bin/env python3
"""
Script to filter TensorBoard log directories based on a pattern.
Can either keep directories matching the pattern or archive them.
"""

import os
import sys
import shutil
from datetime import datetime
import fire
import re

def filter_tb_log(pattern, input_dir="logs/tensorboard", archive_dir="logs/tb_archived", mode="keep"):
    """
    Filter TensorBoard log directories based on the specified pattern.
    
    Args:
        pattern (str): Pattern to match against directory names
        input_dir (str): Path to input TensorBoard logs directory
        archive_dir (str): Path to archive directory for moved logs
        mode (str): Either "keep" (keep matching dirs) or "remove" (archive matching dirs)
    """
    
    if mode not in ["keep", "remove"]:
        print(f"Error: mode must be either 'keep' or 'remove', got '{mode}'")
        return False
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return False
    
    # Create archive directory if it doesn't exist
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir, exist_ok=True)
        print(f"Created archive directory: {archive_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Get all subdirectories in the input directory
        subdirs = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
        
        if not subdirs:
            print(f"No subdirectories found in '{input_dir}'")
            return True
        
        matching_dirs = []
        non_matching_dirs = []
        
        # Separate directories based on whether they match the pattern
        for dirname in subdirs:
            if re.search(pattern, dirname):
                matching_dirs.append(dirname)
            else:
                non_matching_dirs.append(dirname)
        
        print(f"Found {len(subdirs)} total directories in '{input_dir}'")
        print(f"Directories matching pattern '{pattern}': {len(matching_dirs)}")
        print(f"Directories not matching pattern: {len(non_matching_dirs)}")
        
        if mode == "keep":
            # Keep matching directories, archive non-matching ones
            dirs_to_archive = non_matching_dirs
            dirs_to_keep = matching_dirs
            print(f"\nMode: KEEP - Keeping {len(dirs_to_keep)} matching directories")
            print(f"Archiving {len(dirs_to_archive)} non-matching directories")
        else:  # mode == "remove"
            # Archive matching directories, keep non-matching ones
            dirs_to_archive = matching_dirs
            dirs_to_keep = non_matching_dirs
            print(f"\nMode: REMOVE - Archiving {len(dirs_to_archive)} matching directories")
            print(f"Keeping {len(dirs_to_keep)} non-matching directories")
        
        if not dirs_to_archive:
            print("No directories to archive. Operation completed.")
            return True
        
        # Create timestamped subdirectory in archive
        archive_subdir = os.path.join(archive_dir, f"archived_{timestamp}")
        os.makedirs(archive_subdir, exist_ok=True)
        
        archived_count = 0
        failed_count = 0
        
        # Archive directories
        for dirname in dirs_to_archive:
            src_path = os.path.join(input_dir, dirname)
            dst_path = os.path.join(archive_subdir, dirname)
            
            try:
                shutil.move(src_path, dst_path)
                print(f"Archived: {dirname}")
                archived_count += 1
            except Exception as e:
                print(f"Failed to archive '{dirname}': {e}")
                failed_count += 1
        
        # Print summary
        print(f"\n=== Summary ===")
        print(f"Pattern: '{pattern}'")
        print(f"Mode: {mode.upper()}")
        print(f"Total directories processed: {len(subdirs)}")
        print(f"Directories kept in '{input_dir}': {len(dirs_to_keep)}")
        print(f"Directories archived: {archived_count}")
        if failed_count > 0:
            print(f"Failed operations: {failed_count}")
        print(f"Archive location: {archive_subdir}")
        
        if dirs_to_keep:
            print(f"\nDirectories remaining in '{input_dir}':")
            for dirname in sorted(dirs_to_keep):
                print(f"  - {dirname}")
        
        return failed_count == 0
        
    except Exception as e:
        print(f"Error processing directories: {e}")
        return False

if __name__ == "__main__":
    fire.Fire(filter_tb_log)
