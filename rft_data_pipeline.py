"""
Generate the Rejection sampling or 1-step expert iteration dataset from the pset-messages-140k dataset.
by default sample one, using default temperature 1.0. 
Filter the pset-messages-140k dataset if the input model is not able to solve the problem in pass@k.
For the successful problems, sample 1 from them and save the model output messages.

default template name: dskpv2-non-cot

This script:
1. Loads the pset-messages-140k dataset (from local file or HuggingFace Hub)
2. Uses the input model to attempt solving each problem with pass@k
3. Filters out problems that the model CANNOT solve (keeps only solvable ones)
4. For successful problems, randomly samples 1 attempt and adds model output to messages
5. Outputs the filtered dataset as a HuggingFace Dataset
"""

import re
import json
import asyncio
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.utils import extract_code, DEEPSEEK_HEADER, extract_code_from_prq
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
from datasets import load_dataset, Dataset
import fire
import random


async def compile_codes_with_server(queries, lean_server_host, lean_server_port, proofaug, pa_with_orig, hammer_list, require_reconstruct, step_timeout, total_timeout, non_repl):
    """
    Use lean_reward_server for code compilation via HTTP requests
    """
    server_url = f"http://{lean_server_host}:{lean_server_port}"
    # Prepare request data in the format expected by lean_reward_server
    request_data = RewardRequest(
        queries=queries,  # Send codes as queries in completion mode
        proofaug=proofaug,
        pa_with_orig=pa_with_orig,
        hammer_list=hammer_list,
        require_reconstruct=require_reconstruct,
        step_timeout=step_timeout,
        total_timeout=total_timeout,
        non_repl=non_repl,
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


def main(
    input_path="Vivacem/pset-messages-140k",
    model_path="checkpoints/0811-q2515bi-pset10k-sft/", 
    output_dir="data/rft_pset_0811-q2515bi",
    split="train",
    n=1,
    gpu=4,
    max_size=None,
    huggingface_dataset=False,
    template_name="dskpv2-non-cot.json",
    tokenizer=None,
    use_remote_llm=False,
    max_requests_llm=16,
    chat_template_fp=None,
    base_url=None,
    api_key=None,
    max_tokens=None,
    estimate_max_tokens=False,
    max_model_len=4096,
    seed=1,
    temperature=1.0,
    top_p=0.95,
    gpu_memory_utilization=0.9,
    lean_server_host='localhost',
    lean_server_port=5000,
    step_timeout=None,
    total_timeout=None,
    proofaug=False,
    pa_with_orig=False,
    require_reconstruct=True,
    non_repl=False,
    hammer_type=None,
    hammer_list=None,
    hammer_recipe=None,
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the dataset
    data_list = []
    
    # Check if input_path is a Hugging Face dataset or a local file
    if huggingface_dataset or (not os.path.exists(input_path) and not input_path.endswith(('.json', '.jsonl'))):
        # Load from Hugging Face
        logger.info(f"Loading dataset from Hugging Face: {input_path}")
        dataset = load_dataset(input_path, split=split)
        # Convert to list and apply max_size limit
        for i, data in enumerate(dataset):
            data_list.append(data)
            if max_size and len(data_list) >= max_size:
                logger.info(f"Debug mode: limiting to {max_size} problems")
                break
    else:
        # Load from local file
        logger.info(f"Loading dataset from local file: {input_path}")
        with open(input_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                data_list.append(data)
                if max_size and len(data_list) >= max_size:
                    logger.info(f"Debug mode: limiting to {max_size} problems")
                    break
    
    logger.info(f"Loaded {len(data_list)} problems from {input_path}")
    
    # Step 2: Prepare inputs using dskpv2 template and messages field
    model_inputs = []
    prefixes = []
    tokenizer_path = tokenizer if tokenizer else model_path
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Load dskpv2-non-cot template (default template name as per docstring)
    
    with open(join("templates", template_name), mode='r') as f:
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
        if chat_template_fp:
            with open(chat_template_fp, 'r') as f:
                chat_template_content = json.load(f)
        else:
            chat_template_content = None
        
        text = tokenizer_obj.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template_content
        )
        model_inputs.append(text)
        
        # Set prefix for code extraction
        prefixes.append(f"{header}{formal_statement}".split(":= by")[0])
    
    # Step 3: Generate solutions with pass@k
    max_input_tokens = max([len(tokenizer_obj.encode(input_text)) for input_text in model_inputs])
    
    if estimate_max_tokens:
        max_model_len_value = max_model_len
        max_tokens_value = max_model_len_value - max_input_tokens
        logger.info(f"{max_model_len_value=}, {max_input_tokens=}, {estimate_max_tokens=} so we set {max_tokens_value=}")
    else:
        max_tokens_value = max_tokens
        max_model_len_value = max_tokens_value + max_input_tokens
        logger.info(f"{max_tokens_value=}, {max_input_tokens=}, {estimate_max_tokens=} so we set {max_model_len_value=}")
    
    if use_remote_llm:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        e = ThreadPoolExecutor(max_workers=max_requests_llm)
        kwargs = {
            "model": model_path,
            "max_tokens": max_tokens_value,
            "temperature": temperature,
            "n": n,
        }
        if top_p > 0:
            kwargs["top_p"] = top_p
        
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
                model_outputs[idx] = ["Request failed."] * n
    else:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens_value, top_p=top_p, n=n)
        model = LLM(model=model_path, seed=seed, trust_remote_code=True, swap_space=8, tensor_parallel_size=gpu, max_model_len=max_model_len_value, gpu_memory_utilization=gpu_memory_utilization)
        vllm_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)
        model_outputs = [[vllm_output.outputs[i].text for i in range(n)] for vllm_output in vllm_outputs]
        del model
        torch.cuda.empty_cache()
    
    logger.info(f"Generated {len(model_outputs)} sets of solutions")
    
    # Step 4: Extract and compile codes
    to_inference_codes = []
    for i in range(len(data_list)):
        data_list[i]["messages"] = messages_list[i]  # Always use messages format
        data_list[i]["model_outputs"] = model_outputs[i]
        
        full_codes = []
        prompt = model_inputs[i]
        for response in model_outputs[i]:
            full_code = extract_code_from_prq(prompt, response)
            full_codes.append(full_code)
        data_list[i]["full_code"] = full_codes        
        
        name = data_list[i].get("problem_id", data_list[i].get("name"))
        for j, code in enumerate(full_codes):
            to_inference_codes.append({"name": name, "code": code, "data_index": i, "attempt": j})
    
    # Compile all codes
    codes = [code["code"] for code in to_inference_codes]
    logger.info(f"Compiling {len(codes)} codes")
    
    # Set up hammer list
    hammer_list_final = None
    if hammer_recipe:
        hammer_list_final = RECIPE2HAMMER_LIST[hammer_recipe]
    elif hammer_list:
        hammer_list_final = hammer_list
    elif hammer_type:
        hammer_list_final = [hammer_type]
    
    queries = [f"```lean4\n{code}\n```" if code else "" for code in codes]
    outputs_list = asyncio.run(compile_codes_with_server(queries, lean_server_host, lean_server_port, proofaug, pa_with_orig, hammer_list_final, require_reconstruct, step_timeout, total_timeout, non_repl))
    
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]
    
    # Step 5: Group results by problem and check pass@8
    problem_results = {}
    for result in to_inference_codes:
        name = result["name"]
        if name not in problem_results:
            problem_results[name] = []
        problem_results[name].append(result)
    
    # Step 6: Filter problems - keep only those that PASSED pass@k and sample 1 from successful attempts
    filtered_data = []
    total_problems = len(data_list)
    unsolvable_count = 0
    
    for i, data in enumerate(data_list):
        name = data.get("problem_id", data.get("name"))
        if name in problem_results:
            # Check if this problem passes pass@k test
            if check_pass_at_k(problem_results[name], k=n):
                # Find successful attempts
                successful_attempts = [result for result in problem_results[name] if result["compilation_result"]["complete"]]
                if successful_attempts:
                    # Sample 1 from successful attempts
                    chosen_attempt = random.choice(successful_attempts)
                    attempt_idx = chosen_attempt["attempt"]
                    
                    # Update messages to include the model output for the chosen attempt
                    model_response = data["model_outputs"][attempt_idx]
                    updated_messages = data["messages"] + [{"role": "assistant", "content": model_response}]
                    data["messages"] = updated_messages
                    
                    # Keep this problem - model can solve it
                    filtered_data.append(data)
                    logger.debug(f"Problem {name} is solvable - KEEPING (chose attempt {attempt_idx})")
                else:
                    unsolvable_count += 1
                    logger.info(f"Problem {name} passed but no successful attempts found - FILTERING OUT")
            else:
                unsolvable_count += 1
                logger.info(f"Problem {name} is unsolvable - FILTERING OUT")
        else:
            # No results for this problem (likely all attempts failed) - filter it out
            unsolvable_count += 1
            logger.debug(f"Problem {name} had no valid attempts - FILTERING OUT")
    
    # Step 7: Create HuggingFace dataset
    filtered_count = len(filtered_data)
    logger.info(f"RFT dataset: {filtered_count}/{total_problems} problems kept (model can solve), {unsolvable_count} unsolvable problems filtered out")
    
    # Create and save HuggingFace dataset
    if filtered_data:
        rft_dataset = Dataset.from_list(filtered_data)
        dataset_path = os.path.join(output_dir, 'rft_dataset')
        rft_dataset.save_to_disk(dataset_path)
        logger.info(f"RFT dataset saved to: {dataset_path}")
    else:
        logger.warning("No problems passed the filter - no dataset created")
        dataset_path = None
    
    # Save compilation results for analysis
    compilation_path = os.path.join(output_dir, 'compilation_results.json')
    with open(compilation_path, 'w') as f:
        json.dump(to_inference_codes, f, indent=4)
    
    # Save summary
    is_huggingface = huggingface_dataset or (not os.path.exists(input_path) and not input_path.endswith(('.json', '.jsonl')))
    summary = {
        "model": model_path,
        "input_path": input_path,
        "dataset_source": "huggingface" if is_huggingface else "local_file",
        "split": split,
        "total_problems": total_problems,
        "unsolvable_problems": unsolvable_count,
        "filtered_problems": filtered_count,
        "pass_at_k": n,
        "max_size_debug": max_size,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_path": dataset_path,
        "compilation_path": compilation_path
    }
    
    summary_path = os.path.join(output_dir, 'rft_pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"RFT data pipeline complete!")
    if max_size:
        print(f"DEBUG MODE: Limited to {max_size} problems")
    print(f"Total problems: {total_problems}")
    print(f"Unsolvable problems (filtered out): {unsolvable_count}")
    print(f"Solvable problems (kept): {filtered_count}")
    if dataset_path:
        print(f"RFT dataset saved to: {dataset_path}")
    print(f"Summary saved to: {summary_path}")
    
    return rft_dataset if filtered_data else None


if __name__ == "__main__":
    fire.Fire(main)
