import re
import json
import argparse
import asyncio
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code, get_semi_proofs
from prover.logger import logger
import random
LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

async def compile_codes(codes, cpu, memory_limit, timeout=300, ast=False, tactics=False, 
        use_pty=False, pty_restart_count=3, random_order=False):
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=cpu, timeout=timeout, memory_limit=memory_limit, name='verifier', use_pty=use_pty, pty_restart_count=pty_restart_count)
    tasks = [{
            "code": code,
            "ast": ast,
            "tactics": tactics
        } for code in codes]
    indexed_tasks = list(enumerate(tasks))
    if random_order:
        random.shuffle(indexed_tasks)
    indices, shuffled_tasks = zip(*indexed_tasks) if indexed_tasks else ([], [])
    try:
        request_id_list = lean4_scheduler.submit_all_request(shuffled_tasks)
        outputs_list = await lean4_scheduler.async_get_all_request_outputs(request_id_list)
        if random_order:
            output_map = {idx: output for idx, output in zip(indices, outputs_list)}
            outputs_list = [output_map[i] for i in range(len(codes))]
        return outputs_list
    except Exception as e:
        logger.error(f"Error compiling codes: {e}")
        raise e
    finally:
        lean4_scheduler.close()

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
    # Create output directory first
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_records_path = os.path.join(args.output_dir, 'full_records.jsonl')
    
    if args.use_existing_code:
        print(f"Using existing code from {args.use_existing_code}")
        to_inference_codes = []
        with open(args.use_existing_code, 'r') as file:
            for line in file:
                data = json.loads(line)
                name = data["problem_id"] if "problem_id" in data else data["name"]
                to_inference_codes += [{"name": name, "code": code} for code in data["full_code"]]
    elif os.path.exists(full_records_path):
        print(f"Loading existing records from {full_records_path}")
        to_inference_codes = []
        with open(full_records_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                name = data["problem_id"] if "problem_id" in data else data["name"]
                to_inference_codes += [{"name": name, "code": code} for code in data["full_code"]]
    else:
        # Step 1: Inference
        data_list = []
        with open(args.input_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if args.split is None or (data["split"] == args.split):
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
            
            with open(full_records_path, 'a') as f:
                json.dump(data_list[i], f)
                f.write('\n')

    # Step 2: Compile
    if args.proofaug:
        df= pd.DataFrame(to_inference_codes)
        # get a dict that maps name to list of codes
        name_to_codes = df.groupby("name")["code"].apply(list).to_dict()
        to_inference_codes = []
        for name, codes in name_to_codes.items():
            extended_codes = set()
            for code in codes:
                semi_proofs = get_semi_proofs(code, block_threshold=10)
                if args.hammer_type in ['smt', 'hint', 'my_hint']:
                    hint_dict = {
                        'smt': r"smt",
                        'hint': r"hint",
                        'my_hint': r"try norm_num [*]; try field_simp [*] at *; try ring_nf at *; try nlinarith"
                    }
                    omni_tactic = hint_dict[args.hammer_type]
                    subst_proofs = [code.replace('sorry', omni_tactic) for code in semi_proofs]
                    extended_codes.update(subst_proofs)
                elif args.hammer_type == 'smt+aster':
                    raise NotImplementedError("smt+aster i.e. smt [*] is not implemented yet")
                else:
                    raise ValueError(f"Invalid hammer type: {args.hammer_type}")
            to_inference_codes += [{"name": name, "code": code} for code in extended_codes]

    codes = [code["code"] for code in to_inference_codes]
    
    print(f"Compiling {len(codes)} codes")
    outputs_list = asyncio.run(compile_codes(
        codes, args.cpu, args.memory_limit, args.timeout, args.ast, args.tactics, args.use_pty, args.pty_restart_count, args.random_order))
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]

    output_path = f'{args.output_dir}/code_compilation.json'
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(to_inference_codes, json_file, indent=4)

    # Step 3: Summarize
    result, df_grp = summarize_results(to_inference_codes, args.field)
    summary_path = f'{args.output_dir}/compilation_summary.json'
    infos = {
        "model": args.model_path,
        "n": args.n,
        "timestamp": datetime.now().strftime("%m%d-%H%M")
    }
    result.update(infos)
    with open(args.log_file, "a") as f:
        f.write(f"{result}\n")
    with open(summary_path, "w") as f:
        json.dump(result, f)
    
    df_grp.reset_index()[["name", "correct"]].to_csv(summary_path.replace(".json", ".csv"), index=False, header=True, sep='\t', quoting=1, na_rep='Missing')
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-s', '--split', default=None, type=str)
    parser.add_argument('-n', '--n', default=32, type=int)
    parser.add_argument('-c', '--cpu', default=24, type=int)
    parser.add_argument('-g', '--gpu', default=1, type=int)
    parser.add_argument('-f', '--field', default="complete", choices=["complete", "pass"], type=str)
    parser.add_argument('--memory_limit', default=10, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--timeout', default=300, type=int)
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float)
    parser.add_argument('--sync', action='store_true', default=False)
    parser.add_argument('--log_file', default="logs/summary.log", type=str)
    parser.add_argument('--use_existing_code', type=str, default=None)
    parser.add_argument('--hammer_type', type=str, default='my_hint', choices=['smt', 'smt+aster', 'hint', 'my_hint'])
    parser.add_argument('--ast', action='store_true', default=False)
    parser.add_argument('--tactics', action='store_true', default=False)
    parser.add_argument('--use_pty', action='store_true', default=False)
    parser.add_argument('--proofaug', action='store_true', default=False)
    parser.add_argument('--pty_restart_count', default=100, type=int)
    parser.add_argument('--random_order', action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    main(args)