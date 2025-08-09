import re
import json
import argparse
import asyncio
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.utils import extract_code 
from prover.constants import RECIPE2HAMMER_LIST
from prover.logger import logger
import torch
from transformers import AutoTokenizer
from os.path import join
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
import requests  # Add for HTTP requests to lean_reward_server
from prover.agent_utils import RewardRequest
from prover.utils import PROOF_PATTERN, extract_code_from_prq, DEEPSEEK_HEADER

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
    # logger.setLevel(logging.DEBUG)
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_records_path = os.path.join(args.output_dir, 'full_records.jsonl')
    
    if args.use_existing_code:
        print(f"Using existing code from {args.use_existing_code}")
        to_inference_codes = []
        with open(args.use_existing_code, 'r') as file:
            for line in file:
                data = json.loads(line)
                name = data["problem_id"] if "problem_id" in data else data["name"]
                full_codes = [code if code else "" for code in data["full_code"]]
                to_inference_codes += [{"name": name, "code": code} for code in full_codes]
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

        model_inputs = []   # for non-remote
        messages_list = []  # for remote
        prefixes = []
        tokenizer_path = args.tokenizer if args.tokenizer else args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if args.template_name:
            with open(join("templates", args.template_name + ".json"), mode='r') as f:
                template = json.loads(f.read()) # type: dict
            template_examples = []
            if args.template_example: # system, user, assistant, stop
                with open(join("templates", "examples", args.template_example + ".jsonl"), mode='r') as f:
                    for line in f:
                        template_examples.append(json.loads(line))
                template_examples = template_examples[:args.n_shot]
        for data in data_list:
            header = data.get('header', DEEPSEEK_HEADER)
            formal_statement = data['formal_statement'] # until := by\n (or no \n, it depends)
            informal_prefix = data.get('informal_prefix', str())
            if informal_prefix:
                if m:= re.match(r"/--(.*?)--/", informal_prefix.strip(), re.DOTALL):
                    problem = m.group(1)
                else:
                    problem = informal_prefix.strip()
            else:
                problem = '[[Informal problem is not available]]'

            # we provide the following fields:
            # problem, informal_prefix, header, formal statement
            if args.template_name:
                messages = []
                if template.get("system"):
                    messages.append({"role": "system", "content": template["system"]})
                for example in template_examples:
                    messages.append({"role": "user", "content": template["user"].format(**example)})
                    messages.append({"role": "assistant", "content": template["assistant"].format(**example)})
                
                messages.append({"role": "user", "content": template["user"].format(problem=problem, informal_prefix=informal_prefix, header=header, formal_statement=formal_statement)})            
                messages_list.append(messages)
                prefixes.append(f"{header}{formal_statement}".split(":= by")[0])
                # TODO: use model.chat to replace model_inputs
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
                
            else:   # TODO: to be legacy by writing a jinja chat_template for this openrlhf template
                model_inputs.append(f"Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}")
                prefixes.append(f"{header}{informal_prefix}{formal_statement}".split(":=")[0])

        # find the max length of the model input
        def get_num_tokens(text):
            return len(tokenizer.encode(text))
        max_input_tokens = max([get_num_tokens(input) for input in model_inputs])
        if args.estimate_max_tokens:
            max_model_len = args.max_model_len
            max_tokens = max_model_len - max_input_tokens
            logger.info(f"{max_model_len=}, {max_input_tokens=}, {args.estimate_max_tokens=} so we set {max_tokens=}")
        else:
            max_tokens = args.max_tokens
            max_model_len = max_tokens + max_input_tokens
            logger.info(f"{max_tokens=}, {max_input_tokens=}, {args.estimate_max_tokens=} so we set {max_model_len=}")
        
        # generate the model_outputs
        if args.use_remote_llm:
            client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
            e = ThreadPoolExecutor(max_workers=args.max_requests_llm)
            futures = []
            kwargs = {
                "model": args.model_path,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "n": args.n,
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
                    # return response.choices[0].message.content
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
            # TODO: find how can we use the LLM class for chat
            # it seems that model.chat is OK but we need to finish the above
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=max_tokens, top_p=args.top_p, n=args.n)
            model = LLM(model=args.model_path, seed=args.seed, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
            # responses = model.chat(messages_list, sampling_params, use_tqdm=True)
            # model_outputs = [[response.choices[i].message.content for i in range(args.n)] for response in responses]
            vllm_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)
            model_outputs = [[vllm_output.outputs[i].text for i in range(args.n)] for vllm_output in vllm_outputs]
            print(f"example model input:\n{model_inputs[0]}")
            print(f"example model output:\n{model_outputs[0]}")
            del model
            torch.cuda.empty_cache()
        
        to_inference_codes = []
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(len(data_list)):
            data_list[i]["messages"] = messages_list[i] if args.template_name else model_inputs[i]
            data_list[i]["model_outputs"] = model_outputs[i]
            
            full_codes = []
            prompt = model_inputs[i]
            for response in model_outputs[i]:
                full_code = extract_code_from_prq(prompt, response)
                full_codes.append(full_code)
            data_list[i]["full_code"] = full_codes
            name = data_list[i]["problem_id"] if "problem_id" in data_list[i] else data_list[i]["name"]
            to_inference_codes += [{"name": name, "code": code} for code in data_list[i]["full_code"]]
            
            with open(full_records_path, 'a') as f:
                json.dump(data_list[i], f)
                f.write('\n')

    # Step 2: Compile
    codes = [code["code"] for code in to_inference_codes]
    
    print(f"Compiling {len(codes)} codes")

    # determine the hammer_list
    hammer_list = None
    if args.hammer_recipe:
        hammer_list = RECIPE2HAMMER_LIST[args.hammer_recipe]
        if args.hammer_list:
            logger.warning(f"hammer_list is ignored when hammer_recipe is provided")
    elif args.hammer_list:
        hammer_list = args.hammer_list
    elif args.hammer_type:
        hammer_list = [args.hammer_type]
    args.hammer_list = hammer_list

    assert args.use_lean_server, "non-lean_server mode is deprecated"
    queries = [f"```lean4\n{code}\n```" if code else "" for code in codes]
    outputs_list = asyncio.run(compile_codes_with_server(queries, args))
    
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]

    output_path = f'{args.output_dir}/code_compilation.json'
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(to_inference_codes, json_file, indent=4)

    # Step 3: Summarize
    result, df_grp = summarize_results(to_inference_codes, args.field)
    summary_path = f'{args.output_dir}/compilation_summary.json'
    hammers = args.hammer_list if args.hammer_list else [args.hammer_type]
    infos = {
        "model": args.model_path,
        "n": args.n,
        "timestamp": datetime.now().strftime("%m%d-%H%M"),
        "hammers": hammers,
        "output_dir": args.output_dir,
    }
    result.update(infos)
    with open(args.log_file, "a") as f:
        f.write(f"{result}\n")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"{summary_path}")
    df_grp.reset_index()[["name", "correct"]].to_csv(summary_path.replace(".json", ".csv"), index=False, header=True, sep='\t', quoting=1, na_rep='Missing')
    result.update({"compilation_summary": summary_path})
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
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--use_remote_llm', action='store_true', default=False)
    parser.add_argument('--max_requests_llm', default=16, type=int)
    parser.add_argument('--template_name', type=str, default=None)
    parser.add_argument('--template_example', type=str, default=None, help="templates/examples/{}.jsonl")
    parser.add_argument('--chat_template_fp', type=str, default=None)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--base_url', default=None, type=str)
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--max_tokens', default=2048, type=int)
    parser.add_argument('--estimate_max_tokens', action='store_true', default=False, help="when set, use max_model_len to deduce max_tokens, otherwise reversely.")
    parser.add_argument('--max_model_len', default=4096, type=int)
    parser.add_argument('--kimina_prompt', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--memory_limit', default=10, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--step_timeout', default=None, type=int, help="step timeout for the lean server")
    parser.add_argument('--total_timeout', default=None, type=int, help="total timeout for the lean server")
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float)
    parser.add_argument('--sync', action='store_true', default=False)
    parser.add_argument('--log_file', default="results/summary.log", type=str)
    parser.add_argument('--use_existing_code', type=str, default=None)
    parser.add_argument('--ast', action='store_true', default=False)
    parser.add_argument('--tactics', action='store_true', default=False)
    parser.add_argument('--use_pty', action='store_true', default=True)
    parser.add_argument('--nouse_pty', dest='use_pty', action='store_false', default=False)
    parser.add_argument('--hammer_type', type=str, default=None, help="see hint_dict in utils.py for available options")
    parser.add_argument('--hammer_list', nargs='+', default=None)
    parser.add_argument('--hammer_recipe', type=str, default=None)
    parser.add_argument('--proofaug', action='store_true', default=False)
    parser.add_argument('--pa_with_orig', action='store_true', default=False)
    parser.add_argument('--require_reconstruct', action='store_true', default=True)
    parser.add_argument('--proofaug_legacy', action='store_true', default=False)
    parser.add_argument('--pty_restart_count', default=100, type=int)
    parser.add_argument('--use_lean_server', action='store_true', default=True)
    parser.add_argument('--lean_server_host', type=str, default='localhost', help='Lean reward server hostname')
    parser.add_argument('--lean_server_port', type=int, default=5000, help='Lean reward server port')
    parser.add_argument('--non_repl', action='store_true', default=False)


    args = parser.parse_args()
    print(args)
    main(args)