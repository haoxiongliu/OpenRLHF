#!/usr/bin/env python3
"""
Lean Command Generator Web Service
Provides a web interface for the to_command function from prover/utils.py
"""

import uvicorn
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Start the Lean Command Generator web service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure templates directory exists
    templates_dir = os.path.join(current_dir, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    print(f"Starting Lean Command Generator web service...")
    print(f"Current directory: {current_dir}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"API documentation at: http://{args.host}:{args.port}/api/docs")
    
    # Change to the command_generator directory
    os.chdir(current_dir)
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 