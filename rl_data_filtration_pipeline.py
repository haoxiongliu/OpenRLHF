"""
Filter the lean-workbook-messages dataset if Goedel-LM/Goedel-Prover-V2-8B is not able to solve the problem in pass@8.

Adapted from eval_pipeline.py. 
Goedel-Prover-V2-8B uses the dskpv2.json template. DO NOT USE THE DATASET MESSAGES FIELD. USE THE TEMPLATE TO CONSTRUCT MESSAGES.
Support zero-shot is enough. remove the few-shot implementation.

This script:
1. Loads the lean-workbook-messages dataset (from local file or HuggingFace Hub)
2. Uses Goedel-LM/Goedel-Prover-V2-8B to attempt solving each problem with pass@8
3. Filters out problems that the model CAN solve (keeps only unsolvable ones)
4. Saves the filtered dataset for RL training

Unlike eval_pipeline.py, this supports loading datasets directly from HuggingFace Hub
using load_dataset(), which is useful for datasets like "Vivacem/lean-workbook-messages".
"""

import re
import json
import argparse
import asyncio
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.utils import extract_code, DEEPSEEK_HEADER, DEF_SIGN
from prover.constants import RECIPE2HAMMER_LIST
from prover.logger import logger
import torch
from transformers import AutoTokenizer
from os.path import join
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
import requests
from prover.agent_utils import RewardRequest
from datasets import load_dataset


async def compile_codes_with_server(queries, args):
    """
    Use lean_reward_server for code compilation via HTTP requests
    """
    server_url = f"http://{args.lean_server_host}:{args.lean_server_port}"
    # Prepare request data in the format expected by lean_reward_server
    request_data = RewardRequest(
        queries=queries,  # Send codes as queries in completion mode
        proofaug=args.proofaug,
        pa_with_orig=args.pa_with_orig,
        hammer_list=args.hammer_list,
        require_reconstruct=args.require_reconstruct,
        step_timeout=args.step_timeout,
        total_timeout=args.total_timeout,
        non_repl=args.non_repl,
    ).model_dump(exclude_none=True)
    
    logger.info(f"Sending {len(queries)} codes to lean_reward_server at {server_url}")
    response = requests.post(
        f"{server_url}/reward",
        json=request_data,
    )
    response.raise_for_status()
    results = response.json()
    
    # Convert server response to the format expected by eval_pipeline
    outputs_list = []
    for i in range(len(queries)):
        verification_result = {k: v[i] for k, v in results.items()}
        verification_result["complete"] = verification_result["rewards"] > 0
        outputs_list.append(verification_result)
    
    logger.info(f"Received results from lean_reward_server: {sum(results['rewards'])} successful out of {len(queries)}")
    return outputs_list


def check_pass_at_k(results, k=8):
    """
    Check if any of the k attempts succeeded (pass@k)
    """
    return any(result["compilation_result"]["complete"] for result in results[:k])


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load the dataset
    data_list = []
    
    # Check if input_path is a Hugging Face dataset or a local file
    if args.huggingface_dataset or (not os.path.exists(args.input_path) and not args.input_path.endswith(('.json', '.jsonl'))):
        # Load from Hugging Face
        logger.info(f"Loading dataset from Hugging Face: {args.input_path}")
        dataset = load_dataset(args.input_path, split=args.split)
        # Convert to list and apply max_size limit
        for i, data in enumerate(dataset):
            data_list.append(data)
            if args.max_size and len(data_list) >= args.max_size:
                logger.info(f"Debug mode: limiting to {args.max_size} problems")
                break
    else:
        # Load from local file
        logger.info(f"Loading dataset from local file: {args.input_path}")
        with open(args.input_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                data_list.append(data)
                if args.max_size and len(data_list) >= args.max_size:
                    logger.info(f"Debug mode: limiting to {args.max_size} problems")
                    break
    
    logger.info(f"Loaded {len(data_list)} problems from {args.input_path}")
    
    # Step 2: Prepare inputs using dskpv2 template and messages field
    model_inputs = []
    prefixes = []
    tokenizer_path = args.tokenizer if args.tokenizer else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Load dskpv2 template
    with open(join("templates", "dskpv2.json"), mode='r') as f:
        template = json.loads(f.read())
    
    messages_list = []
    for data in data_list:
        # Extract fields needed for template construction
        header = data.get('header', DEEPSEEK_HEADER)
        formal_statement = data['formal_statement']
        informal_prefix = data.get('informal_prefix', str())
        
        # Extract problem from informal_prefix if available
        if informal_prefix:
            if m := re.match(r"/--(.*?)--/", informal_prefix.strip(), re.DOTALL):
                problem = m.group(1)
            else:
                problem = informal_prefix.strip()
        else:
            problem = '[[Informal problem is not available]]'
        
        # Construct messages using dskpv2 template (DO NOT use dataset messages field)
        messages = []
        if template.get("system"):
            messages.append({"role": "system", "content": template["system"]})
        
        # Format the user message using template
        user_content = template["user"].format(
            problem=problem,
            informal_prefix=informal_prefix,
            header=header,
            formal_statement=formal_statement
        )
        messages.append({"role": "user", "content": user_content})
        messages_list.append(messages)
        
        # Apply chat template
        if args.chat_template_fp:
            with open(args.chat_template_fp, 'r') as f:
                chat_template = json.load(f)
        else:
            chat_template = None
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template
        )
        model_inputs.append(text)
        
        # Set prefix for code extraction
        prefixes.append(f"{header}{formal_statement}".split(DEF_SIGN)[0])
    
    # Step 3: Generate solutions with pass@8
    max_input_tokens = max([len(tokenizer.encode(input_text)) for input_text in model_inputs])
    
    if args.estimate_max_tokens:
        max_model_len = args.max_model_len
        max_tokens = max_model_len - max_input_tokens
        logger.info(f"{max_model_len=}, {max_input_tokens=}, {args.estimate_max_tokens=} so we set {max_tokens=}")
    else:
        max_tokens = args.max_tokens
        max_model_len = max_tokens + max_input_tokens
        logger.info(f"{max_tokens=}, {max_input_tokens=}, {args.estimate_max_tokens=} so we set {max_model_len=}")
    
    if args.use_remote_llm:
        client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
        e = ThreadPoolExecutor(max_workers=args.max_requests_llm)
        kwargs = {
            "model": args.model_path,
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "n": args.n,  # Pass@8 requires n=8
        }
        if args.top_p > 0:
            kwargs["top_p"] = args.top_p
        
        def post_request(messages):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    **kwargs
                )
                return [choice.message.content for choice in response.choices]
            except Exception as e:
                logger.error(f"Error posting request: {e}")
                return None
        
        futures = [e.submit(post_request, messages) for messages in messages_list]
        future_to_index = {future: idx for idx, future in enumerate(futures)}
        model_outputs = [None] * len(messages_list)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating model outputs"):
            idx = future_to_index[future]
            response_content = future.result()
            if response_content is not None:
                model_outputs[idx] = response_content
            else:
                model_outputs[idx] = ["Request failed."] * args.n
    else:
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=max_tokens, top_p=args.top_p, n=args.n)
        model = LLM(model=args.model_path, seed=args.seed, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
        vllm_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)
        model_outputs = [[vllm_output.outputs[i].text for i in range(args.n)] for vllm_output in vllm_outputs]
        del model
        torch.cuda.empty_cache()
    
    logger.info(f"Generated {len(model_outputs)} sets of solutions")
    
    # Step 4: Extract and compile codes
    to_inference_codes = []
    for i in range(len(data_list)):
        prefix = prefixes[i]
        full_codes = []
        for output in model_outputs[i]:
            model_code = extract_code(output, strict=args.strict_extract)
            if model_code is None:
                full_code = None
            else:
                mc_prefix_end = model_code.find(DEF_SIGN)
                if mc_prefix_end == -1:
                    logger.debug(f"No {DEF_SIGN=} found in {output}")
                    full_code = None
                else:
                    full_code = prefix + model_code[mc_prefix_end:]
            full_codes.append(full_code)
        
        name = data_list[i].get("problem_id", data_list[i].get("name"))
        for j, code in enumerate(full_codes):
            to_inference_codes.append({"name": name, "code": code, "data_index": i, "attempt": j})
    
    # Compile all codes
    codes = [code["code"] for code in to_inference_codes]
    logger.info(f"Compiling {len(codes)} codes")
    
    # Set up hammer list
    hammer_list = None
    if args.hammer_recipe:
        hammer_list = RECIPE2HAMMER_LIST[args.hammer_recipe]
    elif args.hammer_list:
        hammer_list = args.hammer_list
    elif args.hammer_type:
        hammer_list = [args.hammer_type]
    args.hammer_list = hammer_list
    
    queries = [f"```lean4\n{code}\n```" if code else "" for code in codes]
    outputs_list = asyncio.run(compile_codes_with_server(queries, args))
    
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]
    
    # Step 5: Group results by problem and check pass@8
    problem_results = {}
    for result in to_inference_codes:
        name = result["name"]
        if name not in problem_results:
            problem_results[name] = []
        problem_results[name].append(result)
    
    # Step 6: Filter problems - keep only those that FAILED pass@8
    filtered_data = []
    total_problems = len(data_list)
    solvable_count = 0
    
    for i, data in enumerate(data_list):
        name = data.get("problem_id", data.get("name"))
        if name in problem_results:
            # Check if this problem passes pass@8 test
            if check_pass_at_k(problem_results[name], k=8):
                solvable_count += 1
                logger.info(f"Problem {name} is solvable - FILTERING OUT")
            else:
                # Keep this problem - model cannot solve it
                filtered_data.append(data)
                logger.debug(f"Problem {name} is unsolvable - KEEPING")
        else:
            # No results for this problem (likely all attempts failed) - keep it
            filtered_data.append(data)
            logger.debug(f"Problem {name} had no valid attempts - KEEPING")
    
    # Step 7: Save filtered dataset
    filtered_count = len(filtered_data)
    logger.info(f"Filtered dataset: {filtered_count}/{total_problems} problems kept ({solvable_count} filtered out)")
    
    # Save filtered dataset
    output_path = os.path.join(args.output_dir, 'filtered_dataset.jsonl')
    with open(output_path, 'w') as f:
        for data in filtered_data:
            json.dump(data, f)
            f.write('\n')
    
    # Save compilation results for analysis
    compilation_path = os.path.join(args.output_dir, 'compilation_results.json')
    with open(compilation_path, 'w') as f:
        json.dump(to_inference_codes, f, indent=4)
    
    # Save summary
    is_huggingface = args.huggingface_dataset or (not os.path.exists(args.input_path) and not args.input_path.endswith(('.json', '.jsonl')))
    summary = {
        "model": args.model_path,
        "input_path": args.input_path,
        "dataset_source": "huggingface" if is_huggingface else "local_file",
        "split": args.split,
        "total_problems": total_problems,
        "solvable_problems": solvable_count,
        "filtered_problems": filtered_count,
        "pass_at_k": args.n,
        "max_size_debug": args.max_size,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_path": output_path,
        "compilation_path": compilation_path
    }
    
    summary_path = os.path.join(args.output_dir, 'filtration_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Filtration complete!")
    if args.max_size:
        print(f"DEBUG MODE: Limited to {args.max_size} problems")
    print(f"Total problems: {total_problems}")
    print(f"Solvable problems (filtered out): {solvable_count}")
    print(f"Unsolvable problems (kept): {filtered_count}")
    print(f"Filtered dataset saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter lean-workbook-messages dataset based on Goedel-Prover-V2-8B solvability")
    
    # Required arguments
    parser.add_argument('-i', '--input_path', default="Vivacem/lean-workbook-messages", type=str, help='Path to input dataset (local jsonl file) or HuggingFace dataset name (e.g., "Vivacem/lean-workbook-messages")')
    parser.add_argument('-m', '--model_path', default="Goedel-LM/Goedel-Prover-V2-8B", type=str, help='Path to Goedel-LM/Goedel-Prover-V2-8B model')
    parser.add_argument('-o', '--output_dir', default="data/filtered_debug", type=str, help='Output directory for filtered dataset')
    
    # Optional arguments  
    parser.add_argument('-s', '--split', default="train", type=str, help='Dataset split to process (e.g., "train", "test")')
    parser.add_argument('-n', '--n', default=8, type=int, help='Number of attempts for pass@k (default: 8)')
    parser.add_argument('-g', '--gpu', default=4, type=int, help='Number of GPUs to use')
    parser.add_argument('--max_size', default=None, type=int, help='Debug: maximum number of problems to process (for testing)')
    parser.add_argument('--huggingface_dataset', action='store_true', default=False, help='Force loading from HuggingFace Hub (auto-detected if input_path doesn\'t exist locally)')
    
    # Model arguments
    parser.add_argument('--tokenizer', type=str, default=None, help='Tokenizer path (default: same as model)')
    parser.add_argument('--use_remote_llm', action='store_true', default=False, help='Use remote LLM API')
    parser.add_argument('--max_requests_llm', default=16, type=int, help='Max concurrent requests for remote LLM')
    parser.add_argument('--chat_template_fp', type=str, default=None, help='Path to chat template file')
    parser.add_argument('--base_url', default=None, type=str, help='Base URL for remote LLM API')
    parser.add_argument('--api_key', default=None, type=str, help='API key for remote LLM')
    
    # Generation arguments
    parser.add_argument('--max_tokens', default=None, type=int, help='Maximum tokens to generate')
    parser.add_argument('--estimate_max_tokens', action='store_true', default=False, help='Estimate max tokens from model length')
    parser.add_argument('--max_model_len', default=32768, type=int, help='Maximum model context length')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--temperature', default=1.0, type=float, help='Sampling temperature')
    parser.add_argument('--top_p', default=0.95, type=float, help='Top-p sampling parameter')
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float, help='GPU memory utilization')
    
    # Lean server arguments
    parser.add_argument('--lean_server_host', type=str, default='localhost', help='Lean reward server hostname')
    parser.add_argument('--lean_server_port', type=int, default=5000, help='Lean reward server port')
    parser.add_argument('--step_timeout', default=None, type=int, help='Step timeout for lean server')
    parser.add_argument('--total_timeout', default=None, type=int, help='Total timeout for lean server')
    
    # Proof augmentation arguments
    parser.add_argument('--proofaug', action='store_true', default=False, help='Enable proof augmentation')
    parser.add_argument('--pa_with_orig', action='store_true', default=False, help='Proof augmentation with original')
    parser.add_argument('--require_reconstruct', action='store_true', default=True, help='Require proof reconstruction')
    parser.add_argument('--non_repl', action='store_true', default=False, help='Non-REPL mode')
    
    # Hammer arguments
    parser.add_argument('--hammer_type', type=str, default=None, help='Hammer type to use')
    parser.add_argument('--hammer_list', nargs='+', default=None, help='List of hammers to use')
    parser.add_argument('--hammer_recipe', type=str, default=None, help='Hammer recipe to use')
    
    # Code extraction
    parser.add_argument('--strict_extract', action='store_true', default=False, help='Use strict code extraction')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
