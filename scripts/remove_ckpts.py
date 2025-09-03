#!/usr/bin/env python3
"""
Script to clean up old checkpoints, keeping only the N most recent global_step checkpoints per subdirectory.

Usage examples:
    # Interactive mode (asks questions)
    python scripts/remove_ckpts.py
    
    # Force mode (no questions, delete immediately)
    python scripts/remove_ckpts.py --force
    
    # Keep 5 most recent checkpoints instead of 3
    python scripts/remove_ckpts.py --keep 5 --force
    
    # Dry run only (see what would be deleted)
    python scripts/remove_ckpts.py --dry-run
    
    # Skip dry run and proceed directly
    python scripts/remove_ckpts.py --no-dry-run
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def find_checkpoint_dirs(base_dir: Path) -> List[Tuple[int, Path]]:
    """
    Find all global_step*_hf directories and return them with their step numbers.
    
    Args:
        base_dir: Directory to search in
        
    Returns:
        List of (step_number, path) tuples
    """
    checkpoint_pattern = re.compile(r'^global_step(\d+)_hf$')
    checkpoints = []
    
    for item in base_dir.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, item))
    
    return checkpoints


def clean_checkpoints_in_dir(subdir: Path, keep_count: int = 3, dry_run: bool = False, force: bool = False) -> None:
    """
    Clean checkpoints in a single subdirectory.
    
    Args:
        subdir: Subdirectory to clean
        keep_count: Number of recent checkpoints to keep
        dry_run: If True, only show what would be deleted without actually deleting
        force: If True, skip confirmation prompts
    """
    print(f"Processing directory: {subdir}")
    
    # Find all checkpoint directories
    checkpoints = find_checkpoint_dirs(subdir)
    
    if not checkpoints:
        print("  No global_step checkpoints found")
        return
    
    print(f"  Found {len(checkpoints)} checkpoints")
    
    if len(checkpoints) <= keep_count:
        print(f"  Only {len(checkpoints)} checkpoints found, keeping all")
        return
    
    # Sort by step number (descending - most recent first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    # Split into keep and delete lists
    to_keep = checkpoints[:keep_count]
    to_delete = checkpoints[keep_count:]
    
    print(f"  Keeping {keep_count} most recent checkpoints:")
    for i, (step_num, path) in enumerate(to_keep):
        print(f"    {i+1}. global_step{step_num}_hf")
    
    print(f"  Will {'simulate deleting' if dry_run else 'delete'} {len(to_delete)} old checkpoints:")
    for step_num, path in to_delete:
        print(f"    - global_step{step_num}_hf")
    
    if not to_delete:
        return
    
    # Ask for confirmation if not dry run and not force mode
    if not dry_run and not force:
        response = input(f"  Proceed with deletion? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("  Skipped deletion")
            return
    
    # Delete old checkpoints
    deleted_count = 0
    for step_num, path in to_delete:
        try:
            if dry_run:
                print(f"  [DRY RUN] Would delete: {path}")
            else:
                print(f"  Deleting: {path}")
                shutil.rmtree(path)
            deleted_count += 1
        except Exception as e:
            print(f"  Error deleting {path}: {e}")
    
    print(f"  {'Simulated' if dry_run else 'Deleted'} {deleted_count} old checkpoints")
    print()


def main():
    """Main function to clean up checkpoints."""
    parser = argparse.ArgumentParser(
        description="Clean up old checkpoints, keeping only the N most recent global_step checkpoints per subdirectory."
    )
    parser.add_argument(
        "--keep", "-k", 
        type=int, 
        default=3, 
        help="Number of recent checkpoints to keep (default: 3)"
    )
    parser.add_argument(
        "--force", "-f", 
        action="store_true", 
        help="Skip all confirmation prompts"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--no-dry-run", 
        action="store_true", 
        help="Skip dry run prompt and proceed directly"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="checkpoints/ckpts",
        help="Checkpoint directory path (default: checkpoints/ckpts)"
    )
    
    args = parser.parse_args()
    
    # Configuration
    CHECKPOINT_BASE_DIR = Path(args.dir)
    KEEP_COUNT = args.keep
    
    print("Checkpoint cleanup script")
    print(f"Working directory: {Path.cwd()}")
    print(f"Checkpoint directory: {CHECKPOINT_BASE_DIR.resolve()}")
    print(f"Keeping the {KEEP_COUNT} most recent checkpoints per subdirectory")
    print()
    
    # Check if base directory exists
    if not CHECKPOINT_BASE_DIR.exists():
        print(f"Error: Checkpoint directory {CHECKPOINT_BASE_DIR.resolve()} does not exist!")
        print("Please run this script from the OpenRLHF project root directory.")
        return
    
    # Determine dry run mode
    if args.dry_run:
        dry_run = True
    elif args.no_dry_run:
        dry_run = False
    elif args.force:
        dry_run = False  # Force mode implies no dry run
    else:
        # Ask for dry run only if not in force mode
        dry_run_response = input("Do you want to do a dry run first (see what would be deleted)? (Y/n): ").strip().lower()
        dry_run = dry_run_response not in ['n', 'no']
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be deleted ===\n")
    else:
        print("\n=== LIVE MODE - Files will be permanently deleted ===\n")
    
    # Find all subdirectories (excluding hf_models and hidden directories)
    subdirs = []
    for item in CHECKPOINT_BASE_DIR.iterdir():
        if (item.is_dir() and 
            not item.name.startswith('.') and 
            item.name != 'hf_models'):
            subdirs.append(item)
    
    print(f"Found {len(subdirs)} subdirectories to process")
    print()
    
    # Process each subdirectory
    for subdir in sorted(subdirs):
        clean_checkpoints_in_dir(subdir, KEEP_COUNT, dry_run, args.force)
    
    print("Checkpoint cleanup completed!")
    
    if dry_run:
        print("\nThis was a dry run. Run the script again with --no-dry-run to actually delete files.")


if __name__ == "__main__":
    main()
