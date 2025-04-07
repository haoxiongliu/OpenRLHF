#!/bin/bash
#SBATCH --job-name=multi_node_training
#SBATCH --partition=q-ndfl4zki
#SBATCH --nodes=1               # 使用4个节点
#SBATCH --ntasks=1    # 每个节点运行1个任务
#SBATCH --cpus-per-task=16      # 每个任务使用16个CPU核心
#SBATCH --gres=gpu:8            # 每个节点使用8个GPU
#SBATCH --time=2400:00:00         # 最大运行时间
#SBATCH --output=logs/debug.log  # 输出日志文件
#SBATCH --error=logs/debug.log    # 错误日志文件
#SBATCH --open-mode=append

sleep 360000