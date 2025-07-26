"""Utils that used by agent_func_proofaug.py
Set independent since that module will be dynamically loaded in training.
"""
from pydantic import BaseModel
import re

def remove_indent(content: str) -> str:
    l_indent = len(content.split('\n')[0])
    return "\n".join([line[l_indent:] for line in content.split("\n")])

def add_indent(content: str, indent: int) -> str:
    return "\n".join([" "*indent + line for line in content.split("\n")])

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

def split_header_body(code, remove_comments=True):
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

class RewardRequest(BaseModel):
    """
    This class is used to send request to lean_reward_server.
    """
    queries: list[str]  # in fact prompt+response
    prompts: list[str] | None = None  # in fact prompt only
    labels: list[str] | None = None
    proofaug: bool = False
    hammer_list: list[str|None] | str | None = None
    hammer_recipe: str | None = None
    step_timeout: float | None = None
    total_timeout: float | None = None
    require_reconstruct: bool = False
    pa_with_orig: bool = False
    non_repl: bool = False
    time_reward_ratio: float = 0.0
    time_reward_threshold: float = 120.0


class RewardResponse(BaseModel):
    """
    when RewardResponse(**dict) receive extra fields, it will be ignored.
    """
    rewards: list[float]
    bodies: list[str | None] = [None]
    headers: list[str | None] = [None]
    proofaug_subst: list[dict | None] = [None]
    proofaug_codes: list[str | None] = [None]
    success_types: list[str | None] = [None]
    verify_times: list[float | None] = [None]
    errorss: list[list[str]] = [[]]

if __name__ == "__main__":
    ret_dict = {
        "rewards": [0.0],
        "bodies": ["```lean4\n\n```"],
        "headers": ["```lean4\n\n```"],
        "proofaug_subst": ["```lean4\n\n```"],
        "proofaug_codes": ["```lean4\n\n```"],
        "success_types": ["proofaug"],
        "verify_times": [0.0],
        "errorss": ["error"],
    }
    response = RewardResponse(**ret_dict)
    print(response)