# codes adapted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git
# all copyright to https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git
import os
import json
import pytz
from pathlib import Path
from datetime import datetime
from collections import UserDict
from importlib.machinery import SourceFileLoader
from easydict import EasyDict as AttrDict
from copy import deepcopy
import re
import pandas as pd

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'
LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def non_cot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def non_cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )

def cot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )

def post_process_output(output):
    _find_idx = output.find("```")
    return output[:_find_idx] if _find_idx >= 0 else output

MODEL_FORMAT = dict(
    non_cot=dict(prompt=non_cot_prompt, output=post_process_output, few_shot=non_cot_few_shot_prompt),
    cot=dict(prompt=cot_prompt, output=post_process_output, few_shot=cot_few_shot_prompt),
)


def get_datetime(readable=False):
    if readable:
        return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y/%m/%d %H:%M:%S")
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")

def load_config(fname):
    name = Path(fname).stem
    mod = SourceFileLoader(name, fname).load_module()

    config = {}
    for n in dir(mod):
        if not n.startswith("__"):
            config[n] = getattr(mod, n)
    config = AttrDict(config)

    return config

def load_jsonl_objects(input_path):
    objects = []
    with open(input_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            objects.append(json.loads(line))
    return objects


def extract_code(text: str, strict: bool = False) -> str:
    code = None
    if m:= re.search(r'```lean4\n(.*?)\n```', text, re.DOTALL):
        code = m.group(1)   
    elif m:= re.search(r'```\S*\n(.*)', text, re.DOTALL):
        if strict:
            code = "[[Code no end.]]"
        else:
            code = m.group(1) # no ``` case
    else:
        code = "[[No code found.]]"
    return code

class ConcurrentJob(object):
    def __init__(self, stage_list):
        assert len(stage_list) > 1
        self.stage_list = stage_list
        self.reset()
    
    def is_idle(self):
        return self._stage_idx is None
    
    def reset(self):
        self._stage_idx = None
        self._stage_cache = None
    
    def start(self, **kwargs):
        self._stage_idx = 1
        self._stage_cache = self.stage_list[0](**kwargs)
    
    def get_status(self):
        assert not self.is_idle()
        while True:
            status = self.stage_list[self._stage_idx](**self._stage_cache)
            if status is None:
                return None
            self._stage_idx += 1
            if self._stage_idx == len(self.stage_list):
                self.reset()
                return status
            self._stage_cache = status


def remove_lean_comments(code: str, normalize: bool = False) -> str:
    """
    Remove all Lean comments from the given code.
    This function removes both single-line comments (starting with '--')
    and block comments delimited by '/-' and '-/'.
    If normalize is set, make theorem ... :=  into one line.
    """
    # Remove block comments (which may span multiple lines)
    code_wo_comment = re.sub(r'/\-.*?\-/', '', code, flags=re.DOTALL)
    # Remove single-line comments
    code_wo_comment = re.sub(r'--.*', '', code_wo_comment)
    if normalize:
        # Remove all newlines between 'theorem' and ':='
        # First, find all 'theorem' declarations
        theorem_pattern = re.compile(r'(theorem\s+.*?)(:=)', re.DOTALL)
        
        def replace_newlines(match):
            # Replace all newlines with spaces in the matched group
            theorem_text = match.group(1)
            theorem_text = re.sub(r'\n\s*', ' ', theorem_text)
            return theorem_text + match.group(2)
        
        # Apply the replacement
        code_wo_comment = theorem_pattern.sub(replace_newlines, code_wo_comment)

    # Clean up: strip trailing spaces and remove any empty lines
    lines = [line.rstrip() for line in code_wo_comment.splitlines()]


    return "\n".join(line for line in lines if line.strip())

# TODO: make it legacy. do not rely on line analysis
def is_statement(line: str) -> bool:
    return line.strip().startswith("have") or line.startswith("theorem")


def find_blocks(code: str) -> list[tuple[int, int]]:
    """
    Find 'have..by' blocks in Lean code. theorem..by is also included.
    Recursively finds all blocks, including nested ones and one-line blocks.
    
    Args:
        code: The Lean code as a string
    
    Returns:
        List of tuples containing (start_line, end_line) positions for each block,
        where end_line is the last line of the block.
    """
    # TODO: accodomate  have ...\n := by\n linarith
    blocks = []
    lines = code.splitlines()
    
    def process_lines(start_idx, end_idx, parent_indent=None):
        if end_idx == start_idx + 1:
            return
        i = start_idx
        while i < end_idx:
            line = lines[i]
            stripped_line = line.lstrip()
            current_indent = len(line) - len(stripped_line)
            # Look for lines that start with "have" or "theorem" and contain "by"
            if is_statement(stripped_line):
                start_line = i
                block_indent = current_indent
                i += 1
                
                # Find the end of this block
                while i < end_idx:
                    next_line = lines[i]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= block_indent:
                        break
                    i += 1
                
                end_line = i-1  # last line of the block
                blocks.append((start_line, end_line))
                process_lines(start_line + 1, end_line, block_indent)
                continue
            else:
                i += 1
    process_lines(0, len(lines))
    
    # Sort blocks by start line for consistent output
    return sorted(blocks)
    
def get_semi_proofs(result: dict | str, block_threshold: int = 10) -> list[str]:
    """
    Get all semi-proofs in the given code_completion results.
    For the structure of result, refer to verifier.py
    We can simply find all 'have' keywords or 'by' keywords as the blocks of proof-qed in Isar. Use indentation to determine the end of a block.

    Current version do not rely on errors to infer. Just get all semi-proofs.
    By semi-proof, we mean substituting 

    """
    if isinstance(result, str):
        code = result
    else:
        name, code, cr = result.get('name', ''), result.get('code', ''), result.get('compilation_result', {})
        errors, complete = cr.get('errors', []), cr.get('complete', False)
        assert code, "must have code"
        if complete:
            return [code]
    
    code = remove_lean_comments(code, normalize=True)
    # 1. find the have...by... blocks, end by indentation
    blocks = find_blocks(code)
    # find all maximal non-overlapping block combinations
    def get_maximal_combinations(blocks: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """
        Given a list of blocks as (start, end) tuples (with end included),
        return all maximal combinations of non-overlapping blocks.
        A combination is considered maximal if no additional block from the list
        (beyond those already chosen) can be appended without overlapping.
        """
        results = []
        # Check if a combination is maximal (no more blocks can be added)
        def is_maximal(combination):
            for block in blocks:
                if block not in combination:
                    # Check if this block can be added to the combination
                    can_add = True
                    for comb_block in combination:
                        if not (block[0] > comb_block[1] or block[1] < comb_block[0]):
                            can_add = False
                            break
                    if can_add:
                        return False  # Not maximal if we can add another block
            return True

        def backtrack(start: int, current: list[tuple[int, int]]) -> None:
            # Check if current combination is maximal before adding to results
            if start >= len(blocks):
                if current and is_maximal(current):
                    results.append(current.copy())
                return
                
            block = blocks[start]
            # Option 1: Skip this block
            backtrack(start + 1, current)
            
            # Option 2: Include this block if it doesn't overlap with any block in current
            can_include = True
            for comb_block in current:
                # Check for overlap (end is inclusive, so strict inequality)
                if not (block[0] > comb_block[1] or block[1] < comb_block[0]):
                    can_include = False
                    break
                    
            if can_include:
                current.append(block)
                backtrack(start + 1, current)
                current.pop()  # Backtrack
        
        backtrack(0, [])
        return results

    # Get all maximal non-overlapping block combinations from the found blocks
    if len(blocks) < block_threshold:
        maximal_combinations = get_maximal_combinations(blocks)
    else:
        maximal_combinations = [[blocks[0]]]

    semi_proofs = []
    for combination in maximal_combinations:
        lines = code.splitlines()
        modified_lines = deepcopy(lines)  # Create a copy to modify
        # Process blocks in reverse order to prevent line number shifts
        for start, end in sorted(combination, key=lambda x: x[0], reverse=True):
            original_line = modified_lines[start]
            # need check
            modified_line = re.sub(r'(.*?\bby\b).*', r'\1 sorry', original_line)
            modified_lines[start] = modified_line
            del modified_lines[start+1 : end+1]
        semi_proof = '\n'.join(modified_lines)
        semi_proofs.append(semi_proof)
    
    return semi_proofs



def compare_compilation_summaries(
        pa_path="results/minif2f/DeepSeek-Prover-V1.5-RL-n1-pa", 
        ref_path="results/minif2f/DeepSeek-Prover-V1.5-RL-n1"
    ):
    """served for current version of proofaug 0329"""
    # Load the two compilation summary CSV files
    summary_pa = pd.read_csv(f'{pa_path}/compilation_summary.csv', delimiter='\t')
    summary_f2f = pd.read_csv(f'{ref_path}/compilation_summary.csv', delimiter='\t')

    # Merge the two dataframes on the 'name' column
    merged_summary = pd.merge(summary_pa, summary_f2f, on='name', suffixes=('_pa', '_f2f'))

    # Find the pa correct but f2f incorrect
    merged_summary['difference'] = (merged_summary['correct_pa'] > 0) & (merged_summary['correct_f2f'] == 0)

    # Filter the results to show only the differences
    differences = merged_summary['difference']
    print(f"{pa_path} correct but {ref_path} incorrect:\n {differences.sum()}")
    return differences

def smt_aster(semi_proof: str) -> str:
    """
    Replace sorrries with smt [h0, h1, ...]
    """
    hypo_candidates = ["h" + str(i) for i in range(10)] + ["h" + chr(0x2080 + i) for i in range(10)]
    code = deepcopy(semi_proof)
    while True:
        idx = code.find('sorry')
        if idx == -1:
            break
        hypo_list = [hypo for hypo in hypo_candidates if hypo in code[:idx]]
        tactic = "smt"
        if len(hypo_list) > 0:
            hypos = " ,".join(hypo_list)
            tactic = "smt [" + hypos + "]"
        code = code[:idx] + tactic + code[idx+len('sorry'):]

    return code

