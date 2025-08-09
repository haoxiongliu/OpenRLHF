"""Generate SFT data for the given model and input data. Modified from eval_pipeline.py
given an input message template (such as templates/kimina.json)
to finish these two fields messages you need formal_statement, informal_statement, informal_prefix, 


Then given a messages template, which is different from eval pipeline template.
for example, templates/dskpv2-non-cot.json.

The pipeline is as follows:
1. load seed dataset like Vivacem/pset-messages-140k
2. sample 20000 subset and use kimina-1.7b and kimina template to generate content
3. collect these content into messages and upload to huggingface
No need to compile the code. Please remove these parts.
Now do not support prompt field. One may wrap a new script to generate prompt according to the message content


The current implementation is ad-hoc to Vivacem/pset-messages-140k
"""

import re
import json
import argparse
import os
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.utils import extract_code, extract_code_from_prq, merge_code, split_header_body, PROOF_START, PROOF_PATTERN
from prover.logger import logger
import torch
from transformers import AutoTokenizer
from os.path import join
from tqdm import tqdm
import datasets
from datasets import Dataset
import random


def load_and_sample_dataset(dataset_name, sample_size=20000, split='train') -> Dataset:
    """Load dataset and sample a subset"""
    logger.info(f"Loading dataset {dataset_name}")
    dataset = datasets.load_dataset(dataset_name, split=split)
    
    if len(dataset) > sample_size:
        # Random sampling
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset

def upload_to_huggingface(dataset, repo_name, token=None):
    """Upload dataset to HuggingFace Hub"""
    try:
        dataset.push_to_hub(repo_name, token=token)
        logger.info(f"Successfully uploaded dataset to {repo_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        return False


def main(args):
    """Main pipeline for SFT data generation"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and sample dataset
    logger.info("Starting SFT data generation pipeline")
    seed_ds = load_and_sample_dataset(args.dataset_name, args.sample_size, args.split)
    
    logger.info(f"Processing {len(seed_ds)} examples")
    
    # Step 2: Prepare messages using template
    template_path = join("templates", args.template_name + ".json")
    messages_list = []
    
    # Load template examples if provided
    template_examples = None
    if args.template_example:
        template_examples = []
        example_path = join("templates", "examples", args.template_example + ".jsonl")
        if os.path.exists(example_path):
            with open(example_path, 'r') as f:
                for line in f:
                    template_examples.append(json.loads(line))
            template_examples = template_examples[:args.n_shot]
    
    logger.info("Preparing messages from template")
    with open(template_path, 'r') as f:
        template = json.load(f)
    for i, data_item in tqdm(enumerate(seed_ds), desc="Preparing messages"):
        context = {
            "problem": data_item['problem'],
            "informal_prefix": data_item['informal_prefix'],
            "header": data_item["header"],   # previous version uses different formats.
            "formal_statement": data_item['formal_statement'],
        }
        messages = []
        if "system" in template and template["system"]:
            messages.append({"role": "system", "content": template["system"]})
        if template_examples:   # extra fields including thinking and code
            for example in template_examples:
                messages.append({"role": "user", "content": template["user"].format(**example)})
                messages.append({"role": "assistant", "content": template["assistant"].format(**example)})
        user_content = template["user"].format(**context)
        messages.append({"role": "user", "content": user_content})
        messages_list.append(messages)
    
    # Step 3: Generate model outputs
    logger.info("Generating model outputs")
    if not tokenizer_path:
        tokenizer_path = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Prepare inputs for vLLM
    model_inputs = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs.append(text)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        n=args.n,
        seed=args.seed
    )
    
    # Initialize vLLM model
    model = LLM(
        model=args.model_path,
        seed=args.seed,
        trust_remote_code=True,
        swap_space=8,
        tensor_parallel_size=args.gpu,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Generate outputs. TODO: use dataset.map?
    logger.info(f"Generating outputs for {len(model_inputs)} inputs")
    vllm_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)
    
    # Extract generated text
    model_outputs = []
    for vllm_output in vllm_outputs:
        outputs = [output.text for output in vllm_output.outputs]
        model_outputs.append(outputs)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    
    # Step 4: Create SFT dataset
    logger.info("Creating SFT dataset")
    # sft_data, sft_dataset = create_sft_dataset(seed_ds, model_outputs, args)

    sft_data = []
    output_template_path = join("templates", args.output_template_name + ".json")
    with open(output_template_path, 'r') as f:
        output_template = json.load(f)
    
    count_success = 0
    for i in tqdm(range(len(model_outputs)), desc="Creating SFT dataset"):
        prompt = model_inputs[i]
        for response in model_outputs[i]:
            query = prompt + response
            full_code = extract_code_from_prq(query, prompt, response)
            if not full_code:
                continue
            m_thinking = re.match(r"<think>(.*?)</think>", response, re.DOTALL)
            thinking = m_thinking.group(1).strip() if m_thinking else ""
            context = {
                "problem": seed_ds[i]['problem'],
                "header": seed_ds[i]['header'],
                "informal_prefix": seed_ds[i]['informal_prefix'],
                "formal_statement": seed_ds[i]['formal_statement'],  # in pset-messages it is in fact full code. we only need theorem starts
                "thinking": thinking if not args.remove_think else "",
                "code": full_code
            }
            # Create final messages array including assistant response
            final_messages = []
            if output_template.get("system"):
                final_messages.append({"role": "system", "content": output_template["system"]})
            final_messages.append({"role": "user", "content": output_template["user"].format(**context)})
            final_messages.append({"role": "assistant", "content": output_template["assistant"].format(**context)})
            
            # Create SFT example
            sft_example = {
                'messages': final_messages,
                'problem_id': seed_ds[i].get('problem_id', f'{i}'),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'generated_by': args.model_path,
                }
            }
            sft_example.update(context)
            if args.save_assistant_response:
                sft_example['metadata'].update({'assistant_response': response})
            
            sft_data.append(sft_example)
            count_success += 1
            if args.expect_size and count_success >= args.expect_size:
                break
    
    logger.info(f"Created {count_success} SFT examples from {len(model_outputs)} model outputs")
    
    # Create HuggingFace dataset directly
    hf_dataset = Dataset.from_list(sft_data)
    
    # Save to JSONL
    jsonl_path = os.path.join(args.output_dir, 'train.jsonl')
    hf_dataset.to_json(jsonl_path, orient='records', lines=True)

    
    # Step 5: Upload to HuggingFace (optional)
    if args.upload_to_hf and args.hf_repo_name:
        logger.info("Uploading to HuggingFace")
        upload_to_huggingface(hf_dataset, args.hf_repo_name, args.hf_token)
    
    summary = {
        'total_examples': len(sft_data),
        'original_dataset_size': len(seed_ds),
        'outputs_per_example': args.n,
        'model': args.model_path,
        'template': args.template_name,
        'timestamp': datetime.now().isoformat(),
        'output_dir': args.output_dir
    }
    
    logger.info("SFT data generation completed")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data for Lean theorem proving")

    # Output settings
    parser.add_argument('--output_dir', type=str, default="./results/sft/kimina1.7B-debug",
                        help='Output directory for generated data')
    parser.add_argument('--save_assistant_response', action='store_true', default=False,
                        help='Save assistant response to a file')

    # Dataset settings
    parser.add_argument('--dataset_name', type=str, default='Vivacem/pset-messages',
                        help='Dataset name to load from HuggingFace')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of examples to sample from dataset')
    parser.add_argument('--expect_size', type=int, default=None,
                        help='Expected size of the dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    
    # Model settings
    parser.add_argument('--model_path', type=str, default="AI-MO/Kimina-Prover-Distill-1.7B",
                        help='Path to the model for generation')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer (defaults to model_path)')
    
    # Generation settings
    parser.add_argument('--output_template_name', type=str, default="sft-chat-non-cot",
                        help='Output template name for formatting final messages')
    parser.add_argument('--template_name', type=str, default="kimina",
                        help='Template name to use (e.g., kimina)')
    parser.add_argument('--template_example', type=str, default=None,
                        help='Template example file name')
    parser.add_argument('--n_shot', type=int, default=0,
                        help='Number of few-shot examples to include')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--n', type=int, default=1,
                        help='Number of outputs per input')
    parser.add_argument('--remove_think', action='store_true',
                        help='Remove <think> </think> part from the output')

    
    # Hardware settings
    parser.add_argument('--gpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    
    # HuggingFace upload settings
    parser.add_argument('--upload_to_hf', action='store_true',
                        help='Upload dataset to HuggingFace Hub')
    parser.add_argument('--hf_repo_name', type=str, default=None,
                        help='HuggingFace repository name for upload')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API token')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.upload_to_hf and not args.hf_repo_name:
        raise ValueError("--hf_repo_name is required when --upload_to_hf is set")
    
    print("SFT Data Pipeline Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    main(args)