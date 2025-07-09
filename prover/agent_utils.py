"""Utils that used by agent_func_proofaug.py
Set independent since that module will be dynamically loaded in training.
"""
from pydantic import BaseModel
from typing import List, Dict, Optional

def remove_indent(content: str) -> str:
    l_indent = len(content.split('\n')[0])
    return "\n".join([line[l_indent:] for line in content.split("\n")])

def add_indent(content: str, indent: int) -> str:
    return "\n".join([" "*indent + line for line in content.split("\n")])

class RewardRequest(BaseModel):
    """
    This class is used to send request to lean_reward_server.
    """
    queries: List[str]  # in fact prompt+response
    prompts: Optional[List[str]] = None  # in fact prompt only
    labels: Optional[List[str]] = None
    proofaug: bool = False
    hammer_list: Optional[List[str]|str] = None
    hammer_recipe: Optional[str] = None
    step_timeout: Optional[int] = None
    total_timeout: Optional[int] = None
    require_reconstruct: bool = False
    pa_with_orig: bool = False
    non_repl: bool = False
    time_reward_ratio: float = 0.0
    time_reward_threshold: int = 120


class RewardResponse(BaseModel):
    """
    when RewardResponse(**dict) receive extra fields, it will be ignored.
    """
    rewards: List[float] = [0.0]
    bodies: List[str] = [None]
    proofaug_subst: List[Dict[str, str]] = [{}]
    proofaug_codes: List[str] = [None]
    success_types: List[str] = [None]
    verify_times: List[float] = [None]
    errorss: List[str] = [None]

if __name__ == "__main__":
    ret_dict = {
        "rewards": [0.0],
        "bodies": ["```lean4\n\n```"],
        "proofaug_index": [0],
        "proofaug_ranges": [(0, 0)],
        "proofaug_subst": ["```lean4\n\n```"],
        "proofaug_codes": ["```lean4\n\n```"],
        "success_types": ["proofaug"],
        "verify_times": [0.0],
        "errorss": ["error"],
    }
    response = RewardResponse(**ret_dict)
    print(response)