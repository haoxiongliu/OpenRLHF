"""Utils that used by agent_func_proofaug.py
Set independent since that module will be dynamically loaded in training.
"""
from pydantic import BaseModel

def remove_indent(content: str) -> str:
    l_indent = len(content.split('\n')[0])
    return "\n".join([line[l_indent:] for line in content.split("\n")])

def add_indent(content: str, indent: int) -> str:
    return "\n".join([" "*indent + line for line in content.split("\n")])

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
    orig_rewards: list[float | None] = [None]
    bodies: list[str | None] = [None]
    headers: list[str | None] = [None]
    proofaug_substs: list[dict | None] = [None]
    proofaug_codes: list[str | None] = [None]
    success_types: list[str | None] = [None]
    verify_times: list[float | None] = [None]
    pa_depths: list[int | None] = [None]
    depths: list[int | None] = [None]
    errorss: list[list[str]] = [[]]

if __name__ == "__main__":
    ret_dict = {
        "rewards": [0.0],
        "bodies": ["```lean4\n\n```"],
        "headers": ["```lean4\n\n```"],
        "proofaug_substs": ["```lean4\n\n```"],
        "proofaug_codes": ["```lean4\n\n```"],
        "success_types": ["proofaug"],
        "verify_times": [0.0],
        "errorss": ["error"],
    }
    response = RewardResponse(**ret_dict)
    print(response)