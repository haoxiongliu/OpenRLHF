import re
import json
import argparse
import asyncio
import os
import pandas as pd
from vllm import LLM, SamplingParams
from prover.lean.verifier import Lean4ServerScheduler

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

async def compile_codes(codes, cpu):
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=cpu, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([code["code"] for code in codes])
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
    # Step 1: Inference
    data_list = []
    with open(args.input_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if args.split == "none" or (data["split"] == args.split):
                data_list.append(data)

    model_inputs = []
    for data in data_list:
        format_str = "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}"
        model_inputs.append(format_str.format(
            header=data.get('header', LEAN4_DEFAULT_HEADER),
            informal_prefix=data.get('informal_prefix', str()),
            formal_statement=data['formal_statement'],
        ))

    model = LLM(model=args.model_path, seed=1, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=4096)

    sampling_params = SamplingParams(temperature=1.0, max_tokens=2048, top_p=0.95, n=args.n)
    model_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)

    to_inference_codes = []
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(len(data_list)):
        data_list[i]["model_input"] = model_inputs[i]
        data_list[i]["model_outputs"] = [output.text for output in model_outputs[i].outputs]
        data_list[i]["full_code"] = [re.search(r'```lean4\n(.*?)\n```', model_inputs[i] + output.text, re.DOTALL).group(1) for output in model_outputs[i].outputs]
        to_inference_codes += [{"name": data_list[i]["problem_id"] if "problem_id" in data_list[i] else data_list[i]["name"], "code": code} for code in data_list[i]["full_code"]]
        
        with open(f'{args.output_dir}/full_records.jsonl', 'a') as f:
            json.dump(data_list[i], f)
            f.write('\n')

    # Step 2: Compile
    outputs_list = asyncio.run(compile_codes(to_inference_codes, args.cpu))
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]
    
    output_path = f'{args.output_dir}/code_compilation.json'
    with open(output_path, 'w') as json_file:
        json.dump(to_inference_codes, json_file, indent=4)

    # Step 3: Summarize
    result, df_grp = summarize_results(to_inference_codes, args.field)
    summary_path = f'{args.output_dir}/compilation_summary.json'
    with open(summary_path, "w") as f:
        json.dump(result, f)

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
    args = parser.parse_args()
    
    main(args) 