#!/usr/bin/env python3
import os
import psutil
import torch
import pynvml
import logging
from typing import List, Optional
import argparse
import time

logger = logging.getLogger(__name__)

def get_gpu_memory_usage():
    """Get current GPU memory usage for all available GPUs."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_usage = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_usage.append({
            'total': info.total / 1024**2,  # Convert to MB
            'used': info.used / 1024**2,
            'free': info.free / 1024**2
        })
    
    pynvml.nvmlShutdown()
    return memory_usage

def get_gpu_processes():
    """Get list of processes using GPU."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_processes = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for proc in processes:
            try:
                process = psutil.Process(proc.pid)
                gpu_processes.append({
                    'pid': proc.pid,
                    'name': process.name(),
                    'gpu_id': i,
                    'memory_used': proc.usedGpuMemory / 1024**2 if proc.usedGpuMemory else 0,  # Convert to MB
                    'status': process.status()
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    pynvml.nvmlShutdown()
    return gpu_processes

def cleanup_gpu_memory():
    """Clean up GPU memory by synchronizing CUDA and emptying cache."""
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")
    except Exception as e:
        logger.error(f"Error cleaning up GPU memory: {str(e)}")

def detect_zombie_processes(threshold_mb: float = 100.0) -> List[dict]:
    """
    Detect potential zombie processes that are using GPU memory but not showing up in nvidia-smi.
    
    Args:
        threshold_mb: Memory threshold in MB to consider a process as potentially zombie
        
    Returns:
        List of dictionaries containing information about potential zombie processes
    """
    zombie_processes = []
    
    # Get all Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Get GPU processes from nvidia-smi
    gpu_processes = get_gpu_processes()
    gpu_pids = {p['pid'] for p in gpu_processes}
    
    # Check for Python processes using significant memory but not in nvidia-smi
    for proc in python_processes:
        try:
            memory_mb = proc.memory_info().rss / 1024**2
            if memory_mb > threshold_mb and proc.pid not in gpu_pids:
                zombie_processes.append({
                    'pid': proc.pid,
                    'name': proc.name(),
                    'memory_mb': memory_mb,
                    'status': proc.status()
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return zombie_processes

def force_cleanup_gpu_memory():
    """
    Force cleanup of GPU memory by:
    1. Synchronizing CUDA
    2. Emptying cache
    3. Detecting and logging zombie processes
    """
    cleanup_gpu_memory()
    
    # Log current GPU memory usage
    memory_usage = get_gpu_memory_usage()
    for i, usage in enumerate(memory_usage):
        logger.info(f"GPU {i} Memory Usage: {usage['used']:.2f}MB used, {usage['free']:.2f}MB free")
    
    # Detect and log zombie processes
    zombie_processes = detect_zombie_processes()
    if zombie_processes:
        logger.warning("Potential zombie processes detected:")
        for proc in zombie_processes:
            logger.warning(f"PID: {proc['pid']}, Name: {proc['name']}, Memory: {proc['memory_mb']:.2f}MB, Status: {proc['status']}")
    else:
        logger.info("No zombie processes detected") 

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='GPU Memory Monitor and Cleanup Tool')
    parser.add_argument('--interval', type=int, default=5,
                      help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--threshold', type=float, default=100.0,
                      help='Memory threshold in MB to consider a process as zombie (default: 100.0)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--cleanup', action='store_true',
                      help='Perform cleanup after monitoring')
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        while True:
            logger.info("\n=== GPU Memory Status ===")
            
            # Get GPU memory usage
            memory_usage = get_gpu_memory_usage()
            for i, usage in enumerate(memory_usage):
                logger.info(f"GPU {i}:")
                logger.info(f"  Total: {usage['total']:.2f}MB")
                logger.info(f"  Used:  {usage['used']:.2f}MB")
                logger.info(f"  Free:  {usage['free']:.2f}MB")
            
            # Get GPU processes
            gpu_processes = get_gpu_processes()
            if gpu_processes:
                logger.info("\n=== GPU Processes ===")
                for proc in gpu_processes:
                    logger.info(f"PID: {proc['pid']}, Name: {proc['name']}, "
                              f"GPU: {proc['gpu_id']}, Memory: {proc['memory_used']:.2f}MB, "
                              f"Status: {proc['status']}")
            
            # Detect zombie processes
            zombie_processes = detect_zombie_processes(threshold_mb=args.threshold)
            if zombie_processes:
                logger.warning("\n=== Potential Zombie Processes ===")
                for proc in zombie_processes:
                    logger.warning(f"PID: {proc['pid']}, Name: {proc['name']}, "
                                 f"Memory: {proc['memory_mb']:.2f}MB, Status: {proc['status']}")
            
            if args.cleanup:
                logger.info("\n=== Performing Cleanup ===")
                force_cleanup_gpu_memory()
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")

if __name__ == '__main__':
    main() 