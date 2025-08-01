# codes adapted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git
# all copyright to https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git
import os
import json
import pytz
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from easydict import EasyDict as AttrDict
from copy import deepcopy
import re
import random
import pandas as pd
# from datasets import load_dataset, Dataset
import glob
import csv
from typing import Optional
from os.path import join

DEF_SIGN=":="
PROOF_START=":= by"
HOME_DIR = os.path.expanduser('~')
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LAKE_PATH = join(HOME_DIR, '.elan/bin/lake')
DEFAULT_REPL_PATH = join(PROJ_DIR, 'lean-gym-repl/.lake/build/bin/repl')
DEFAULT_LEAN_WORKSPACE= join(PROJ_DIR, 'lean-gym')
DEEPSEEK_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

def non_cot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', DEEPSEEK_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def non_cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', DEEPSEEK_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )

def cot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', DEEPSEEK_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', DEEPSEEK_HEADER),
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

def write_as_jsonl(items: list[dict] | dict, filepath, mode='x', comple_nl=False, verbose=False, ensure_ascii=False):
    # in r+ mode, after read, the write will always start at the end 
    if dirpath:=os.path.dirname(filepath):
        os.makedirs(dirpath, exist_ok=True)
    if 'a' in mode and comple_nl:
        with open(filepath, 'r+') as f:
            if f.read()[-1] != '\n':
                f.write('\n')
    if isinstance(items, dict):
        items = [items]
    with open(filepath, mode) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii)+'\n')
    if verbose:
        print(f'{len(items)} items saved to {filepath}')



def extract_header(code: str) -> str:
    """
    Extract the header from the code.
    """
    keywords = ["have", "theorem", "example", "abbrev", "opaque", "def exercise"]
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if any(line.strip().startswith(kw) for kw in keywords):
            return "\n".join(lines[:i])
    return ""

def merge_code(prompt_code: str, response_code: str) -> str:
    """
    Given a prompt and a response both including ```lean4\n...\n```, return the full code.
    should use the prompt header+statement and response theorem content
    """
    if PROOF_START not in prompt_code or PROOF_START not in response_code:
        return ""
    prefix = prompt_code.split(PROOF_START)[0]
    sep_pos = response_code.find(PROOF_START)  # cannot use split since response may contain multiple DEF_SIGN in the proof
    return prefix + response_code[sep_pos:]

# ybb. just remove all before </think>. if remaining part has <think>, remove all content
def remove_think(text: str) -> str:
    # find the first <think> and last </think> (if no last </think>, remove all content)
    think_end = text.rfind('</think>')
    if think_end != -1:
        text = text[think_end+len('</think>'):]
    if '<think>' in text:
        text = ""
    return text

def extract_code(text: str, strict: bool = False, omit_think: bool = True) -> Optional[str]:
    if omit_think:
        text = remove_think(text)
    code = None
    pattern = r'```lean4?\n(.*?)\n```'
    
    last_match = None
    for match in re.finditer(pattern, text, re.DOTALL):
        last_match = match # Keep updating last_match with the latest match found
    if last_match:
        code = last_match.group(1)
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


# TODO: make it legacy. do not rely on line analysis
def statement_starts(snippet: str) -> bool:
    """starting by Lean definition-like commands"""
    keywords = ["have", "theorem", "example", "abbrev", "opaque", "def exercise"]
    return any(snippet.strip().startswith(kw) for kw in keywords)

def analyzable(snippet: str) -> bool:
    """ending by := by. while := and by not in the same line is valid in Lean, we require in proofaug so."""
    return re.search(r':=\s*by', snippet, re.DOTALL) is not None

def has_statement(code: str) -> bool:
    """Check if the code has any statement."""
    lines = code.splitlines()
    return any(statement_starts(line) for line in lines)

def n_indent(line: str) -> int:
    """Return the number of indentations in the line."""
    return len(line) - len(line.lstrip())

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
            if statement_starts(stripped_line):
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

def remove_lean_comments(code: str, normalize: bool = False) -> str:
    """
    Remove all Lean comments from the given code.
    This function removes both single-line comments (starting with '--')
    and block comments delimited by '/-' and '-/'.
    If normalize is set, make theorem ... :=  into one line.
    """
    # TODO: nested block comments? currently no need.
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

def split_header_body(code: str, remove_comments=True):
    """No strip, just split the code into header and body."""
    # TODO: add support for more keywords, or other heuristics
    # This is ad-hoc for proofnet dataset
    if remove_comments:
        clean_code = remove_lean_comments(code)
        # match = re.search(r'\b(theorem|example|def exercise|def lemma)', clean_code)
    else:
        clean_code = code
    match = re.search(r'(?<=\n)(theorem|example|def exercise|def lemma)', clean_code, re.DOTALL)
    if match is not None:
        header, body = clean_code[:match.start()], clean_code[match.start():]
    else:
        header, body = "", clean_code
    return header, body


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

def compare_compilation_summaries(
        pa_name="DeepSeek-Prover-V1.5-RL-n1-pa", 
        ref_name="DeepSeek-Prover-V1.5-RL-n1"
    ):
    """served for current version of proofaug 0329"""
    # Load the two compilation summary CSV files
    summary_pa = pd.read_csv(f'results/minif2f/{pa_name}/compilation_summary.csv', delimiter='\t')
    summary_f2f = pd.read_csv(f'results/minif2f/{ref_name}/compilation_summary.csv', delimiter='\t')

    # Merge the two dataframes on the 'name' column
    merged_summary = pd.merge(summary_pa, summary_f2f, on='name', suffixes=('_pa', '_f2f'))

    # Find the pa correct but f2f incorrect
    merged_summary['pa_unique_correct'] = (merged_summary['correct_pa'] > 0) & (merged_summary['correct_f2f'] == 0)
    merged_summary['pa_unique_incorrect'] = (merged_summary['correct_pa'] == 0) & (merged_summary['correct_f2f'] > 0)

    # Filter the results to show only the differences
    print(f"{pa_name} correct but {ref_name} incorrect:\n {merged_summary['pa_unique_correct'].sum()}")
    print(f"{pa_name} incorrect but {ref_name} correct:\n {merged_summary['pa_unique_incorrect'].sum()}")

    logs = ""
    # logs include pa unique incorrect list and pa unique correct list
    logs += f"{pa_name} unique incorrect:\n"
    logs += f"{merged_summary[merged_summary['pa_unique_incorrect']]['name'].tolist()}\n"
    logs += f"{pa_name} unique correct:\n"
    logs += f"{merged_summary[merged_summary['pa_unique_correct']]['name'].tolist()}\n"
    
    log_path = f"logs/{pa_name}_{ref_name}_compilation_summary.txt"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(logs)

    return merged_summary

def get_cumulative_pass(
    csv_path_pattern: str = "results/minif2f/Kimina-Prover-Preview-Distill-1.5B-n32-0425-kimina*/compilation_summary.csv"
) -> dict:
    """
    Get the cumulative pass rate of the given csv files using the csv library.
    Returns a dict containing problems that are incorrect in base_path but correct in other files.
    """
    
    # Find all matching CSV files
    csv_paths = glob.glob(csv_path_pattern)
    
    # Find the shortest path as base
    csv_paths = sorted(csv_paths, key=len)
    base_path = csv_paths[0]
    print(f"Base path: {base_path}")
    print(f"Found {len(csv_paths)} CSV files")
    
    # Dictionary to store results: name -> {path -> correct}
    problem_results = {}
    
    # Read all CSV files
    for path in csv_paths:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                name = row['name']
                correct = int(row['correct'])
                
                if name not in problem_results:
                    problem_results[name] = {}
                
                problem_results[name][path] = correct
    
    # Count problems correct in any file
    any_correct = []
    base_incorrect_others_correct = []
    
    for name, results in problem_results.items():
        # Check if correct in any file
        is_correct_anywhere = any(results.get(path, 0) > 0 for path in csv_paths)
        
        # Check if incorrect in base but correct elsewhere
        is_incorrect_in_base = base_path in results and results[base_path] == 0
        is_correct_elsewhere = any(path != base_path and results.get(path, 0) > 0 for path in csv_paths)
        
        if is_correct_anywhere:
            any_correct.append(name)
        
        if is_incorrect_in_base and is_correct_elsewhere:
            base_incorrect_others_correct.append(name)
    
    # Print results
    print("\nProblems incorrect in base_path but correct in other files:")
    for name in base_incorrect_others_correct:
        print(name)
    
    print(f"\nTotal number of such problems: {len(base_incorrect_others_correct)}")
    print(f"Total number of problems correct in at least one file: {len(any_correct)}")
    
    return {
        "base_incorrect_others_correct": base_incorrect_others_correct,
        "any_correct": any_correct,
        "problem_results": problem_results
    }



def result2leanfiles(
    compilation_json: str = 'results/minif2f/Kimina-Prover-Preview-Distill-1.5B-n1-0422/code_compilation.json', 
    output_dir: str = 'mathlib4/MyTest/Kimina-Prover-Preview-Distill-1.5B-n1-0422/',
    strategy: str = 'random'
    ):
    """
    Convert a compilation json to a lean file.
    """
    complete_dir = os.path.join(output_dir, 'complete')
    failed_dir = os.path.join(output_dir, 'failed')
    os.makedirs(complete_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    with open(compilation_json, "r") as f:
        compilation = json.load(f)
    name2complete = defaultdict(list)
    name2failed = defaultdict(list)
    for result in compilation:
        name = result["name"]
        code = result["code"]
        if result["compilation_result"]["complete"]:
            name2complete[name].append(code)
        else:
            name2failed[name].append(code)

    if strategy == 'random':
        for name, codes in name2complete.items():
            with open(os.path.join(complete_dir, f"{name}.lean"), "w") as f:
                f.write(random.choice(codes))
        for name, codes in name2failed.items():
            if name not in name2complete:
                with open(os.path.join(failed_dir, f"{name}.lean"), "w") as f:
                    f.write(random.choice(codes))


def to_command(code, env=None, mode="cmd", proofState=None, sorries=None, verbose=False):
    cmd = {}
    code_key = "cmd" if proofState is None else "tactic"
    cmd[code_key] = code
    if env is not None:
        cmd["env"] = env
    if proofState is not None:
        cmd["proofState"] = proofState
    if sorries is not None: # "grouped" or "individual"
        cmd["sorries"] = sorries
    if verbose:
        print(json.dumps(cmd, ensure_ascii=False))
    return cmd


def has_unrecoverable_error(messages: list[str]) -> bool:
    """
    Check if the messages contain any unrecoverable errors.
    """
    return any(re.search(r"[tT]imeout", message) for message in messages)


# The following functions are used to check the proof status.
# for "linear" branch of repl, v4.19.0.

def compile_errors(result: dict) -> list[str]:
    """handle ['messages'][0]['severity] format and ['message] format of repl,
    only extract the error message due to unable to compile. DOES NOT HANDLE OPEN GOALS REMAIN.
    {"sorries":
    [{"proofState": 1,
    "pos": {"line": 3, "column": 31},
    "parentDecl": "mathd_algebra_263",
    "goals":
    ["y : ℝ\nh₀ : 0 ≤ 19 + 3 * y\nh₁ : √(19 + 3 * y) = 7\n⊢ 0 ≤ 19 + 3 * y"],
    "endPos": {"line": 3, "column": 36}}],
    "messages":
    [{"severity": "error",
    "pos": {"line": 2, "column": 14},
    "endPos": {"line": 3, "column": 36},
    "data":
    "unsolved goals\ny : ℝ\nh₀ : 0 ≤ 19 + 3 * y\nh₁ : √(19 + 3 * y) = 7\nh₁' : 0 ≤ 19 + 3 * y\n⊢ y = 10"}],
    "env": 2}    
    
    {'message': 'Lean error:\n<input>:1:1: unknown tactic'} 
    """
    errors = []
    if "message" in result:
        message = result["message"]
        error_cands = ["error", "unknown proof state", "unknown environment"]
        if any(cand in message.lower() for cand in error_cands):
            errors.append(message)
    elif "messages" in result:
        for message in result["messages"]:
            if message["severity"] == "error" and "unsolved goals" not in message["data"]:
                errors.append(message["data"])
    return errors

def is_incomplete(result: dict) -> bool:
    """
    Check if the proof is incomplete according to the repl result.
    """
    flag = False
    if 'Incomplete: open goals remain' in result.get('message', ''):
        flag = True
    elif "messages" in result:
        for message in result["messages"]:
            if message["severity"] == "error" and "unsolved goals" in message["data"]:
                flag = True
    return flag



def is_complete(result: dict, body: str) -> bool:
    """
    Check if the proof is complete according to the repl result.
    legacy version of lean (<4.20.0) has no "Goals accomplished" in the info.
    we do not care whether there is a compilation error. we only care if all goals are solved.
    """
    # we cannot use extract_errors here since it does not extract 
    if 'proofStatus' in result:
        return result['proofStatus'] == 'Completed'  # linear version of repl
    else:        
        if not has_statement(body):
            return False
        for m in result.get("messages", []):
            if "declaration uses 'sorry'" in m["data"]:
                return False
            if m["severity"] == "error":
                return False
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pa_name", type=str, default="DeepSeek-Prover-V1.5-RL-n1-pa")
    parser.add_argument("--ref_name", type=str, default="DeepSeek-Prover-V1.5-RL-n1")
    args = parser.parse_args()
    compare_compilation_summaries(args.pa_name, args.ref_name)


