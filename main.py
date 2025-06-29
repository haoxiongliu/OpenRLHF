#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple CLI dispatcher that maps command names to functions.
Usage:
    python main.py <command> [--param=value ...] [positional_args...]
"""
import sys
import inspect
import uvicorn
from prover.utils import compare_compilation_summaries, get_cumulative_pass

# Define your commands here
def foo(a, b, flag=False):
    """Example command foo that prints its arguments."""
    print(f"foo called with a={a}, b={b}, flag={flag}")

# Add more command functions as needed
def bar(x=0):
    """Example command bar that prints x."""
    print(f"bar called with x={x}")

def start_generator(host="0.0.0.0", port=8123, reload=False):
    """Start the Lean Command Generator web service with REPL backend."""
    print(f"Starting Lean Command Generator web service...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API documentation at: http://{host}:{port}/api/docs")
    
    uvicorn.run(
        "prover.command_generator.app:app",
        host=host,
        port=int(port),
        reload=bool(reload)
    )

# Mapping from command name to function
commands = {
    'foo': foo,
    'bar': bar,
    'compare_compilation_summaries': compare_compilation_summaries,
    'get_cumulative_pass': get_cumulative_pass,
    'start_generator': start_generator,
}

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <command> [--param=value ...] [positional_args...]")
        print("Available commands:")
        for name, func in commands.items():
            print(f"  {name}: {func.__doc__ or ''}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print("Available commands:", ', '.join(commands.keys()))
        sys.exit(1)

    func = commands[cmd]
    sig = inspect.signature(func)
    params = sig.parameters

    # Parse CLI arguments (named and positional)
    args = []
    kwargs = {}
    it = iter(sys.argv[2:])
    for token in it:
        if token.startswith('--'):
            key_val = token[2:]
            if '=' in key_val:
                name, value = key_val.split('=', 1)
            else:
                name = key_val
                try:
                    value = next(it)
                except StopIteration:
                    print(f"Missing value for argument {name}")
                    sys.exit(1)
            if name not in params:
                print(f"Unknown parameter {name} for command {cmd}")
                sys.exit(1)
            annotation = params[name].annotation
            if annotation is bool:
                v = value.lower()
                value = v == "true"
            elif annotation != inspect._empty:
                value = annotation(value)
            kwargs[name] = value
        else:
            args.append(token)

    # Assign positional args to parameters not already in kwargs
    positional_params = [p for p in params.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    unassigned = [p for p in positional_params if p.name not in kwargs]
    if len(args) > len(unassigned):
        print(f"Too many positional arguments: provided {len(args)}, expected {len(unassigned)}")
        sys.exit(1)
    for value, param in zip(args, unassigned):
        annotation = param.annotation
        if annotation != inspect._empty:
            value = annotation(value)
        kwargs[param.name] = value

    # Execute the command function
    result = func(**kwargs)
    # if result is not None:
    #     print(result)

if __name__ == '__main__':
    main() 