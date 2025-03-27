import re
import json
import argparse
import asyncio
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

async def compile_codes(codes, cpu, memory_limit, timeout=300, ast=False, tactics=False):
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=cpu, timeout=timeout, memory_limit=memory_limit, name='verifier')
    tasks = [{
        "code": code,
        "ast": ast,
        "tactics": tactics
    } for code in codes]
    request_id_list = lean4_scheduler.submit_all_request(tasks)
    outputs_list = await lean4_scheduler.async_get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    return outputs_list

def summarize_results(codes, field):
    df = pd.DataFrame(codes)
    df["correct"] = df.compilation_result.apply(lambda x: x[field])
    df_grp = df.groupby("name")["correct"].sum()
    result = {
        "total": len(df_grp),
        "correct": sum(df_grp > 0),
        "accuracy": f"{sum(df_grp > 0) / len(df_grp) * 100:.2f}",
        "field": field
    }
    return result, df_grp


def main(args):
    if not args.use_existing_code:
        # Step 1: Inference
        data_list = []
        with open(args.input_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if args.split == "none" or (data["split"] == args.split):
                    data_list.append(data)

        model_inputs = []
        for data in data_list:
            format_str = "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}"
            model_inputs.append(format_str.format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            ))

        model = LLM(model=args.model_path, seed=1, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization)

        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=2048, top_p=0.95, n=args.n)
        model_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)

        to_inference_codes = []
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(len(data_list)):
            data_list[i]["model_input"] = model_inputs[i]
            data_list[i]["model_outputs"] = [output.text for output in model_outputs[i].outputs]
            data_list[i]["full_code"] = [extract_code(model_inputs[i] + output.text, strict=False) for output in model_outputs[i].outputs]
            name = data_list[i]["problem_id"] if "problem_id" in data_list[i] else data_list[i]["name"]
            to_inference_codes += [{"name": name, "code": code} for code in data_list[i]["full_code"]]
            
            with open(f'{args.output_dir}/full_records.jsonl', 'a') as f:
                json.dump(data_list[i], f)
                f.write('\n')
    else:
        print(f"Using existing code from {args.use_existing_code}")
        to_inference_codes = []
        with open(args.use_existing_code, 'r') as file:
            for line in file:
                data = json.loads(line)
                name = data["problem_id"] if "problem_id" in data else data["name"]
                to_inference_codes += [{"name": name, "code": code} for code in data["full_code"]]

    # Step 2: Compile
    outputs_list = asyncio.run(compile_codes(
        to_inference_codes, args.cpu, args.memory_limit, args.timeout, args.ast, args.tactics))
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]

    output_path = f'{args.output_dir}/code_compilation.json'
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(to_inference_codes, json_file, indent=4)

    # Step 3: Summarize
    result, df_grp = summarize_results(to_inference_codes, args.field)
    summary_path = f'{args.output_dir}/compilation_summary.json'
    with open(summary_path, "w") as f:
        json.dump(result, f)
    print(result)
    infos = {
        "model": args.model_path,
        "n": args.n,
        "timestamp": datetime.now().strftime("%m%d-%H%M")
    }
    infos.update(result)
    with open(args.log_file, "a") as f:
        f.write(f"{infos}\n")

    df_grp.reset_index()[["name", "correct"]].to_csv(summary_path.replace(".json", ".csv"), index=False, header=True, sep='\t', quoting=1, na_rep='Missing')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-s', '--split', default=None, type=str)
    parser.add_argument('-n', '--n', default=32, type=int)
    parser.add_argument('-c', '--cpu', default=64, type=int)
    parser.add_argument('-g', '--gpu', default=1, type=int)
    parser.add_argument('-f', '--field', default="complete", choices=["complete", "pass"], type=str)
    parser.add_argument('--memory_limit', default=10, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--timeout', default=300, type=float)
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float)
    parser.add_argument('--sync', action='store_true', default=False)
    parser.add_argument('--log_file', default="logs/summary.log", type=str)
    parser.add_argument('--use_existing_code', type=str, default=None)
    parser.add_argument('--ast', action='store_true', default=False)
    parser.add_argument('--tactics', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    main(args)