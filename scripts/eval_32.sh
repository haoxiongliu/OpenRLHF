#!/bin/bash
# please make sure 4 GPUs are available
# only reserve 24 cpus for it

name=$1 # 0319-n4-nokl
num_gpus=${2:-4} # Default to 4 GPUs if not specified
num_cpus=${3:-48} # Default to 24 CPUs if not specified
n=${4:-32}
memory_util=${5:-0.9}
ckpt_root=checkpoints/ckpts/$name
tgt_root=results/minif2f/$name
for ckpt_dir in $(ls -1d $ckpt_root/*/); do
    # exclude _actor
    tag=$(basename $ckpt_dir)
    if [[ $tag == *_actor* ]]; then
        continue
    fi
    tgt_dir=$tgt_root/$tag-n$n
    # if exist $tgt_dir/compilation_summary.csv, skip
    if [ -f $tgt_dir/compilation_summary.csv ]; then
        echo "skip $name/$tag"
        continue
    fi
    eval_args=(
        -i datasets/minif2f.jsonl
        -m $ckpt_dir
        -o $tgt_dir
        -n $n
        -g $num_gpus
        -c $num_cpus
        -s test
	    --use_pty
        --memory_limit 10
        --timeout 300
        --gpu_memory_utilization $memory_util
    )
    echo "Evaluating $name/$tag for pass@32"
    python eval_pipeline.py "${eval_args[@]}"
done
