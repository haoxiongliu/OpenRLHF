#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple CLI dispatcher using Python Fire.
Usage:
    python main.py <command> [--param=value ...] [positional_args...]
"""
import fire
import uvicorn
import os
from prover.utils import compare_compilation_summaries, get_cumulative_pass

def foo(a, b, flag=False):
    """Example command foo that prints its arguments."""
    print(f"foo called with a={a}, b={b}, flag={flag}")

def bar(x=0):
    """Example command bar that prints x."""
    print(f"bar called with x={x}")

def start_generator(host="0.0.0.0", port=8123, reload=False, lean_workspace=None, repl_path=None, config_name=None):
    """Start the Lean Command Generator web service with REPL backend.
    
    Args:
        host: Host address to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8123)
        reload: Enable auto-reload for development (default: False)
        lean_workspace: Optional path to lean workspace directory
    """
    print(f"Starting Lean Command Generator web service...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API documentation at: http://{host}:{port}/api/docs")
    
    if lean_workspace:
        print(f"Using lean workspace: {lean_workspace}")
        # Set environment variable for app.py to pick up
        os.environ['LEAN_WORKSPACE'] = lean_workspace
    if repl_path:
        print(f"Using repl path: {repl_path}")
        os.environ['REPL_PATH'] = repl_path
    uvicorn.run(
        "prover.command_generator.app:app",
        host=host,
        port=int(port),
        reload=bool(reload)
    )

def main():
    fire.Fire()

if __name__ == '__main__':
    main() 