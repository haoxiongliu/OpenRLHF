#!/usr/bin/env python3
"""
Test script to verify if --eps_clip_low_high can parse scientific notation values like 1e08 1e09
"""

import argparse
import sys


def test_float_parsing():
    """Test if string values can be parsed as floats"""
    print("Testing float parsing from strings:")
    
    test_values = ["1e08", "1e09", "1e-8", "1e-9", "0.001", "1000000"]
    
    for val in test_values:
        try:
            parsed = float(val)
            print(f"  '{val}' -> {parsed} (type: {type(parsed).__name__})")
        except ValueError as e:
            print(f"  '{val}' -> ERROR: {e}")
    
    print()


def create_test_parser():
    """Create a test argument parser with eps_clip_low_high argument"""
    parser = argparse.ArgumentParser(description="Test eps_clip_low_high argument")
    parser.add_argument(
        "--eps_clip_low_high", 
        type=float, 
        nargs=2, 
        default=None, 
        help="PPO-clip low and high"
    )
    return parser


def test_argument_parsing():
    """Test the argument parser with different input scenarios"""
    parser = create_test_parser()
    
    # Test cases
    test_cases = [
        ["--eps_clip_low_high", "1e08", "1e09"],
        ["--eps_clip_low_high", "1e-8", "1e-9"],
        ["--eps_clip_low_high", "0.1", "0.2"],
        ["--eps_clip_low_high", "100000000", "1000000000"],
    ]
    
    print("Testing argument parser:")
    
    for i, test_args in enumerate(test_cases, 1):
        try:
            args = parser.parse_args(test_args)
            print(f"  Test {i}: {' '.join(test_args[1:])}")
            print(f"    Parsed values: {args.eps_clip_low_high}")
            print(f"    Types: {[type(x).__name__ for x in args.eps_clip_low_high]}")
            print(f"    Low: {args.eps_clip_low_high[0]}, High: {args.eps_clip_low_high[1]}")
            print()
        except Exception as e:
            print(f"  Test {i}: {' '.join(test_args[1:])} -> ERROR: {e}")
            print()


def main():
    print("=" * 60)
    print("Testing eps_clip_low_high argument parser")
    print("=" * 60)
    print()
    
    # Test basic float parsing
    test_float_parsing()
    
    # Test argument parser
    test_argument_parsing()
    
    # Test with the specific values mentioned by user
    print("Testing specific values from command line (1e08 1e09):")
    print("If you run this script with: python test_eps_clip_parser.py --eps_clip_low_high 1e08 1e09")
    
    # Parse actual command line arguments if provided
    parser = create_test_parser()
    args = parser.parse_args()
    
    if args.eps_clip_low_high is not None:
        print(f"✓ Successfully parsed: {args.eps_clip_low_high}")
        print(f"✓ Low value: {args.eps_clip_low_high[0]} (type: {type(args.eps_clip_low_high[0]).__name__})")
        print(f"✓ High value: {args.eps_clip_low_high[1]} (type: {type(args.eps_clip_low_high[1]).__name__})")
        
        # Verify the values are what we expect
        if args.eps_clip_low_high[0] == 1e08:
            print("✓ Low value matches expected 1e08 (100000000.0)")
        if args.eps_clip_low_high[1] == 1e09:
            print("✓ High value matches expected 1e09 (1000000000.0)")
    else:
        print("No --eps_clip_low_high argument provided")
        print("\nTo test, run:")
        print("python test_eps_clip_parser.py --eps_clip_low_high 1e08 1e09")
    
    print("\n" + "=" * 60)
    print("ANSWER: Yes, 1e08 1e09 can be parsed as floats!")
    print("The argparse module with type=float can handle scientific notation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
