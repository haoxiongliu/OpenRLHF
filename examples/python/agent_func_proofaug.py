"""ProofAug agent function (in implementation)
Use the lean_reward_server.py as the remote reward model.
Not actually an agent. only 1-step, it is just for API compatibility."""
from typing import Any, Dict
import aiohttp

REMOTE_RM_URL = "http://localhost:5000/reward"  # 替换为你的远程奖励模型URL

async def call_remote_reward_model(queries, prompts, labels):
    """async call remote reward model.
    Returns: a dict of contents, including rewards, proofaug_result"""
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
            "labels": labels
        }
        async with aiohttp.client.ClientSession() as session:
            async with session.post(REMOTE_RM_URL, json=data, headers=headers, timeout=60) as response:
                response.raise_for_status()
                result = await response.json()
                return result
                
    except Exception as e:
        print(f"Remote reward model error: {e}")
        return None

async def step(state, action, label, **kwargs) -> Dict[str, Any]:
    """Execute one step of verification and return a random reward using torch.rand

    Args:
        state: The input prompt/expression
        action: The language model's response
        label: Agent identifier or additional information

    Returns:
        Dict[str, Any]: A dictionary containing:
            - rewards: Reward value for advantage calculation
            - scores: Reward value for dynamic filtering
            - next_state: The updated state after the step
            - done: Boolean indicating if the episode is complete
            - sampling_params: Parameters for vLLM sampling
            - extra_logs: Additional logging information
    """
    # TODO: if want to add proofaug, we need to first modify vllm_engine_async.py
    ret_obj = await call_remote_reward_model(state+action, state, label)
    if ret_obj is None:
        reward = 0.0
    else:
        reward = ret_obj["rewards"][0]
    next_state = state + action # will be different for proofaug
    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "next_state": next_state,  # The updated state for vLLM in next step
        "done": True,  # Boolean indicating if the episode is complete
        "sampling_params": kwargs.get("sampling_params", None),  # Parameters for vLLM sampling in next step
        "extra_logs": {"dummy_scores": reward},  # Additional logging information
    }
