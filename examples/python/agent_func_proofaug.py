"""ProofAug agent function (in implementation)
Use the lean_reward_server.py as the remote reward model.
Not actually an agent. only 1-step, it is just for API compatibility."""
from typing import Any, Dict
import aiohttp
import re

REMOTE_RM_URL = "http://localhost:5000/reward"  # 替换为你的远程奖励模型URL

async def call_remote_reward_model(
        queries, prompts, labels, **kwargs):
    """async call remote reward model.
    Returns: a dict of contents, including rewards, proofaug_result"""
    proofaug_config = kwargs.get("proofaug_config") # type: dict
    hammer_list = proofaug_config.get("hammer_list", None)
    hammer_recipe = proofaug_config.get("hammer_recipe", None)
    proofaug = proofaug_config.get("proofaug", False)
    step_timeout = proofaug_config.get("step_timeout", 60)
    remote_timeout = proofaug_config.get("remote_timeout", 300)
    total_timeout = proofaug_config.get("total_timeout", None)
    
    try:
        headers = {"Content-Type": "application/json"}
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(labels, str):
            labels = [labels]
        data = {
            "queries": queries,
            "prompts": prompts, 
            "labels": labels,
            "proofaug": proofaug,
            "hammer_list": hammer_list,
            "hammer_recipe": hammer_recipe,
            "require_reconstruct": True,
            "step_timeout": step_timeout,
            "pa_with_orig": True,
            "total_timeout": total_timeout,
        }
        async with aiohttp.client.ClientSession() as session:
            async with session.post(REMOTE_RM_URL, json=data, headers=headers, timeout=remote_timeout) as response:
                response.raise_for_status()
                result = await response.json()
                return result
                
    except Exception as e:
        print(f"Remote reward model error: {e}")
        return None

async def step(observation, action, label, **kwargs) -> Dict[str, Any]:
    """Execute one step of verification and return a random reward using torch.rand

    Args:
        observation: The input prompt/expression
        action: The language model's response
        label: Agent identifier or additional information
        kwargs: can include proofaug, hammer_list, etc.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - rewards: Reward value for advantage calculation
            - scores: Reward value for dynamic filtering
            - next_observation: The updated observation after the step
            - done: Boolean indicating if the episode is complete
            - sampling_params: Parameters for vLLM sampling
            - extra_logs: Additional logging information
    """
    # TODO: if want to add proofaug, we need to first modify vllm_engine_async.py
    proofaug_config = kwargs.get("proofaug_config", None)
    assert proofaug_config is not None, "proofaug_config is required"
    proofaug = proofaug_config.get("proofaug", False)
    proofaug_ans_subst = proofaug_config.get("proofaug_ans_subst", False)

    ret_obj = await call_remote_reward_model(observation+action, observation, label, **kwargs)
    ret_obj = dict() if ret_obj is None else ret_obj
    reward = ret_obj.get("rewards", [0.0])[0]
    proofaug_code = ret_obj.get("proofaug_codes", [None])[0]
    success_type = ret_obj.get("success_types", [None])[0]

    # find ```lean4 ``` code block in action and replace it with proofaug_proof
    # TODO: use PSA to replace the code blocks in the thinking part
    # this is the most key part!!!
    if proofaug and proofaug_code and success_type == "proofaug" and proofaug_ans_subst:
        think_start = action.find('<think>')
        think_end = action.rfind('</think>')
        
        if think_start != -1 and think_end != -1:
            # Keep think part unchanged, only replace lean4 code blocks outside think part
            before_think = action[:think_start]
            think_part = action[think_start:think_end+len('</think>')]
            # TODO: find the think part ```tactics and find its occuration.
            # This is not definitely helpful.
            after_think = action[think_end+len('</think>'):]
            
            # Replace lean4 code blocks only after_think
            pattern = r'```lean4\s*\n(.*?)\n```'
            def replace_lean4_block(match):
                return f'```lean4\n{proofaug_code}\n```'
            
            after_think_replaced = re.sub(pattern, replace_lean4_block, after_think, flags=re.DOTALL)
            
            ret_action = before_think + think_part + after_think_replaced
        else:
            # No think tags, replace all lean4 code blocks
            pattern = r'```lean4\s*\n(.*?)\n```'
            def replace_lean4_block(match):
                return f'```lean4\n{proofaug_code}\n```'
            
            ret_action = re.sub(pattern, replace_lean4_block, action, flags=re.DOTALL)
    else:
        ret_action = action

    next_observation = observation + ret_action
    # breakpoint()
    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "next_observation": next_observation,  # The updated observation for vLLM in next step
        "done": True,  # Boolean indicating if the episode is complete
        "extra_logs": {},  # Additional logging information
    }
