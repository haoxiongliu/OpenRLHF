"""
Generate the Rejection sampling or 1-step expert iteration dataset from the pset-messages-140k dataset.
by default sample one, using default temperature 1.0. 
Filter the pset-messages-140k dataset if the input model is not able to solve the problem in pass@k.
For the successful problems, sample 1 from them and save the model output messages (append to the messages used to generate model outputs)

default template name: dskpv2-non-cot

This script:
1. Loads the pset-messages-140k dataset (from local file or HuggingFace Hub)
2. Uses the input model to attempt solving each problem with pass@k
3. Filters out problems that the model CANNOT solve (keeps only solvable ones)
4. For successful problems, randomly samples 1 attempt and adds model output to messages used to generate model outputs
5. Outputs the filtered dataset as a HuggingFace Dataset
"""

import re
import json

import os
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
from datasets import load_dataset, Dataset
import fire
import random
from collections import defaultdict
import requests
from prover.agent_utils import RewardResponse, RewardRequest

def compile_codes_with_server(queries, lean_server_host, lean_server_port, proofaug, pa_with_orig, hammer_list, require_reconstruct, step_timeout, total_timeout):
    """
    Use lean_reward_server for code compilation via HTTP requests
    """
    server_url = f"http://{lean_server_host}:{lean_server_port}/reward"
    # Prepare request data in the format expected by lean_reward_server
    request_data = RewardRequest(
        queries=queries,  # Send codes as queries in completion mode
        proofaug=proofaug,
        pa_with_orig=pa_with_orig,
        hammer_list=hammer_list,
        require_reconstruct=require_reconstruct,
        step_timeout=step_timeout,
        total_timeout=total_timeout,
    ).model_dump(exclude_none=True)
    
    response = requests.post(server_url, json=request_data, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    results = response.json()
    results = RewardResponse(**results)

    return results


def check_pass_at_k(results, k=8, only_orig=False):
    """
    Check if any of the k attempts succeeded (pass@k)
    """
    if only_orig:
        any_pass = any(result["success_types"] == "original" for result in results[:k])
    else:
        any_pass = any(result["compilation_result"]["complete"] for result in results[:k])
    return any_pass


def main(
    input_path="Vivacem/pset-messages-140k",    # only support hf now
    model_path="checkpoints/0811-q2515bi-pset10k-sft/", 
    output_dir="results/sft_data/rft_pset_0811-q2515bi",
    orig_hub: str|None = None, # "Vivacem/rft_pset_0811-q2515bi-orig"
    pa_hub: str|None = None, # "Vivacem/rft_pset_0811-q2515bi-pa"
    messages_field: str = "non-cot-messages",
    n=1, max_size=20000, shuffle=False,
    gpu=4, gpu_memory_utilization=0.9,
    template_name="dskpv2-non-cot", tokenizer=None, chat_template_fp=None,
    use_remote_llm=False, max_requests_llm=64, 
    base_url=None, api_key=None, 
    max_tokens=None, max_model_len=4096,
    estimate_max_tokens=True, max_prompt_len=1024,
    seed=1, temperature=1.0, top_p=0.95,
    lean_server_host='localhost', lean_server_port=5000,
    step_timeout=60, total_timeout=180,
    proofaug=True, require_reconstruct=True,
    hammer_type=None, hammer_list=None, hammer_recipe="mix6",
):
    os.makedirs(output_dir, exist_ok=True)
    
    data_list : list[dict] = []
    dataset = load_dataset(input_path, split="train")
    if shuffle:
        dataset = dataset.shuffle()
    to_remove_first = ["messages", "non-cot-messages"]
    to_remove_first = [col for col in to_remove_first if col in dataset.column_names]
    dataset = dataset.remove_columns(to_remove_first)
    data_list = dataset.select(range(min(max_size, len(dataset))))
    logger.info(f"Loaded {len(data_list)} problems from {input_path}")
    
    # Step 2: Prepare inputs using dskpv2 template and messages field
    model_inputs = []
    tokenizer_path = tokenizer if tokenizer else model_path
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    with open(join("templates", template_name + '.json'), mode='r') as f:
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
    
    # Step 3: Generate solutions with pass@k
    max_input_tokens = max([len(tokenizer_obj.encode(input_text)) for input_text in model_inputs])
    if max_prompt_len:
        max_input_tokens = min(max_input_tokens, max_prompt_len)
    
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
    flat_items = []
    for i in range(len(data_list)):
        data_list[i][messages_field] = messages_list[i]  # Always use messages format
        data_list[i]["model_outputs"] = model_outputs[i]
        full_codes = []
        prompt = model_inputs[i]
        for response in model_outputs[i]:
            full_code = extract_code_from_prq(prompt, response)
            full_codes.append(full_code)
        data_list[i]["full_code"] = full_codes        
        name = data_list[i].get("problem_id", data_list[i].get("name"))
        for j, code in enumerate(full_codes):   # flattened
            flat_items.append({"name": name, "code": code, "data_index": i, "attempt": j})
    
    logger.info(f"Compiling {len(flat_items)} codes")
    hammer_list_final = None
    if hammer_recipe:
        hammer_list_final = RECIPE2HAMMER_LIST[hammer_recipe]
    elif hammer_list:
        hammer_list_final = hammer_list
    elif hammer_type:
        hammer_list_final = [hammer_type]
    codes = [item["code"] for item in flat_items]
    queries = [f"```lean4\n{code}\n```" if code else "" for code in codes]
    compile_response = compile_codes_with_server(
        queries=queries, 
        lean_server_host=lean_server_host, 
        lean_server_port=lean_server_port, 
        proofaug=proofaug, 
        pa_with_orig=True, 
        hammer_list=hammer_list_final, 
        require_reconstruct=require_reconstruct, 
        step_timeout=step_timeout, total_timeout=total_timeout
    )
    for i in range(len(flat_items)):
        flat_items[i]["success_type"] = compile_response.success_types[i]
    name2item = defaultdict(list)
    for flat_item in flat_items:
        name = flat_item["name"]
        name2item[name].append(flat_item)
        
    filtered_orig = []
    filtered_pa = []
    total_num = len(data_list)
    for i, data in enumerate(data_list):
        name = data.get("problem_id", data.get("name"))
        data["name"] = name
        assert name in name2item, f"Problem {name} not found in name2item"
        items = name2item[name]

        data_orig = data.copy() # this is a deep copy
        data_pa = data.copy()
        # Find successful attempts
        succ_idx = [item["attempt"] for item in items if item["success_type"] in ["original", "pa_orig", "proofaug"]]
        orig_succ_idx = [item["attempt"] for item in items if item["success_type"] == "original"]
        # Sample 1 from successful attempts
        chosen_orig_idx = chosen_pa_idx = None
        if orig_succ_idx:
            chosen_orig_idx = random.choice(orig_succ_idx)
            data_orig[messages_field] = data[messages_field] + [{"role": "assistant", "content": data['model_outputs'][chosen_orig_idx]}]
            filtered_orig.append(data_orig)
        if succ_idx:
            chosen_pa_idx = chosen_orig_idx if chosen_orig_idx is not None else random.choice(succ_idx)
            data_pa[messages_field] = data[messages_field] + [{"role": "assistant", "content": data['model_outputs'][chosen_pa_idx]}]
            filtered_pa.append(data_pa)

    logger.info(f"RFT dataset: {len(filtered_orig)=}, {len(filtered_pa)=}, {total_num=}")
    
    # Create and save HuggingFace dataset
    rft_pa, rft_orig = Dataset.from_list(filtered_pa), Dataset.from_list(filtered_orig)
    to_remove = ["model_outputs", "full_code", "informal_statement"]
    to_remove = [col for col in to_remove if col in rft_pa.column_names]
    rft_pa = rft_pa.remove_columns(to_remove)
    rft_orig = rft_orig.remove_columns(to_remove)
    pa_path, orig_path = os.path.join(output_dir, 'rft_pa'), os.path.join(output_dir, 'rft_orig')
    rft_pa.save_to_disk(pa_path)
    rft_orig.save_to_disk(orig_path)
    logger.info(f"RFT dataset saved to: {pa_path}")
    logger.info(f"RFT dataset saved to: {orig_path}")
    if orig_hub:
        rft_pa.push_to_hub(pa_hub)
        rft_orig.push_to_hub(orig_hub)
        logger.info(f"RFT dataset pushed to HuggingFace: {pa_hub}")
        logger.info(f"RFT dataset pushed to HuggingFace: {orig_hub}")
    
    # Save compilation results for analysis
    compilation_path = os.path.join(output_dir, 'compilation_results.json')
    with open(compilation_path, 'w') as f:
        json.dump(flat_items, f, indent=4)
    
    # Save summary
    summary = {
        "model": model_path,
        "input_path": input_path,
        "total_num": total_num,
        "filtered_orig": len(filtered_orig),
        "filtered_pa": len(filtered_pa),
        "messages_field": messages_field,
        "pass_at_k": n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pa_output_path": pa_path,
        "orig_output_path": orig_path,
        "pa_hub": pa_hub,
        "orig_hub": orig_hub,
        "compilation_path": compilation_path
    }
    summary_path = os.path.join(output_dir, 'rft_pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"RFT data pipeline complete!")
    logger.info(f"Total problems: {total_num} with {max_size=}")
    logger.info(f"Summary saved to: {summary_path}")
    
    return


if __name__ == "__main__":
    fire.Fire(main)
