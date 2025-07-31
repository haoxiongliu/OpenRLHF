"""ProofAug agent function (in implementation)
Use the lean_reward_server.py as the remote reward model.
Not actually an agent. only 1-step, it is just for API compatibility."""
from typing import Any
import aiohttp
import asyncio
import re
from copy import deepcopy
from prover.agent_utils import RewardResponse, RewardRequest
import logging
from os.path import join

logger = logging.getLogger(__name__)
# set logger file
logger.setLevel(logging.INFO)
handler = logging.FileHandler(join("logs", "agent_func_proofaug.log"))
# set timestamp format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

REMOTE_RM_URL = "http://localhost:5000/reward"  # 替换为你的远程奖励模型URL

async def call_remote_reward_model(
        queries, prompts, labels, **kwargs) -> RewardResponse:
    """async call remote reward model.
    Returns: a RewardResponse object"""
    proofaug_config = kwargs.get("proofaug_config") # type: dict
    hammer_list = proofaug_config.get("hammer_list", None)
    hammer_recipe = proofaug_config.get("hammer_recipe", None)
    proofaug = proofaug_config.get("proofaug", False)
    step_timeout = proofaug_config.get("step_timeout", 60)
    remote_timeout = proofaug_config.get("remote_timeout", 300)
    total_timeout = proofaug_config.get("total_timeout", None)
    
    headers = {"Content-Type": "application/json"}
    if isinstance(queries, str):
        queries = [queries]
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(labels, str):
        labels = [labels]
    data = RewardRequest(
        queries=queries,
        prompts=prompts, 
        labels=labels,
        proofaug=proofaug,
        hammer_list=hammer_list,
        hammer_recipe=hammer_recipe,
        require_reconstruct=True,
        step_timeout=step_timeout,
        pa_with_orig=True,
        total_timeout=total_timeout,
    ).model_dump(exclude_none=True)
    async with aiohttp.ClientSession() as session:
        async with session.post(REMOTE_RM_URL, json=data, headers=headers, timeout=aiohttp.ClientTimeout(total=remote_timeout)) as response:
            response.raise_for_status()
            result = await response.json()
            result = RewardResponse(**result)
    return result


async def step(observation: str, action: str, label: str, **kwargs) -> dict[str, Any]:
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
    subst_rule = proofaug_config.get("subst_rule", None) # keep_depth
    depth_thres = proofaug_config.get("depth_thres", None)
    proofaug_think_mode = proofaug_config.get("proofaug_think_mode", None)
    code_only = proofaug_config.get("code_only", False)
    part_reward = proofaug_config.get("part_reward", 0.5)

    try:
        ret_obj = await call_remote_reward_model(observation+action, observation, label, **kwargs) # type: RewardResponse
    except asyncio.TimeoutError:
        logger.info(f"TimeoutError: {observation+action=}")
        return {"rewards": 0.0, 
                "scores": 0.0, 
                "next_observation": observation + action, 
                "done": True, 
                "extra_logs": {
                    "orig_rewards": 0.0,
                }
            }
    reward = ret_obj.rewards[0]
    orig_reward = ret_obj.orig_rewards[0]
    proofaug_code = ret_obj.proofaug_codes[0]
    success_type = ret_obj.success_types[0]
    header = ret_obj.headers[0]
    body = ret_obj.bodies[0]
    depth = ret_obj.depths[0]
    pa_depth = ret_obj.pa_depths[0]

    if reward > 0.0 and code_only:
        action = f"```lean4\n{header}{body}\n```"

    if proofaug and proofaug_code and success_type == "proofaug" and proofaug_ans_subst:
        from prover.agent_utils import remove_indent
        think_start = action.find('<think>')
        think_end = action.rfind('</think>')
        body = ret_obj.bodies[0]
        proofaug_subst = ret_obj.proofaug_substs[0]
        

        if subst_rule == "maxdepth" and (pa_depth < max(depth_thres, depth)):
            reward = part_reward
            logger.info(f"{subst_rule=}: {pa_depth=} < max({depth_thres}, {depth=}) => keep the original action {action=} rather than using {proofaug_code=}")
            ret_action = action            
        elif subst_rule == "ge2depth" and (pa_depth < min(2, depth)):
            reward = part_reward
            logger.info(f"{subst_rule=}: {pa_depth=} < min(2, {depth=}) => keep the original action {action=} rather than using {proofaug_code=}")
            ret_action = action
        elif subst_rule == "keep_depth" and pa_depth < depth:
            reward = part_reward
            logger.info(f"{subst_rule=}: {pa_depth=} < {depth=} => keep the original action {action=} rather than using {proofaug_code=}")
            ret_action = action
        elif think_start != -1 and think_end != -1 and proofaug_think_mode:
            # Keep think part unchanged, only replace lean4 code blocks outside think part
            before_think = action[:think_start]
            think_part = action[think_start:think_end+len('</think>')]
            modified_think = deepcopy(think_part)   # type: str
            after_think = action[think_end+len('</think>'):]
            block_pattern = r'(?<=```tactics\n).*?(?=\n```)'
            tactic_blocks = re.findall(block_pattern, think_part, re.DOTALL) # type: list[str]

            # note that kimina tactic block can be repeated single tactics
            # should remove all extra thinking after the final tactic block for correct ones
            assert proofaug_think_mode in ["replace_v1", "remove", "remain"], f"Invalid proofaug_think_mode: {proofaug_think_mode}"
            if proofaug_think_mode == "replace_v1":
                for rng, pa_block in proofaug_subst.items():
                    start, end = map(int, rng.split(':'))
                    orig_block = '\n'.join(body.split('\n')[start:end])
                    orig_block_no_indent = remove_indent(orig_block)
                    for i, tactic_block in enumerate(tactic_blocks):
                        if orig_block_no_indent in tactic_block:
                            # breakpoint()
                            modified_think = modified_think.replace(tactic_block, pa_block)
            elif proofaug_think_mode == "remove":
                modified_think = ""
            elif proofaug_think_mode == "remain":
                pass # maintain the original think part
            
            lean4_pattern = r'```lean4\s*\n(.*?)\n```'
            def replace_lean4_block(match):
                return f'```lean4\n{proofaug_code}\n```'
            
            modified_after = re.sub(lean4_pattern, replace_lean4_block, after_think, flags=re.DOTALL)
            
            ret_action = before_think + modified_think + modified_after
            logger.info(f"proofaug modification for {action=} => {ret_action=}")
        else:
            # No think tags, replace all lean4 code blocks
            lean4_pattern = r'```lean4\s*\n(.*?)\n```'
            def replace_lean4_block(match):
                return f'```lean4\n{proofaug_code}\n```'
            
            ret_action = re.sub(lean4_pattern, replace_lean4_block, action, flags=re.DOTALL)
            logger.info(f"proofaug modification for {action=} => {ret_action=}")
    else:
        ret_action = action

    next_observation = observation + ret_action
    # breakpoint()
    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "next_observation": next_observation,  # The updated observation for vLLM in next step
        "done": True,  # Boolean indicating if the episode is complete
        "extra_logs": {
            "orig_rewards": orig_reward,
        },  # Additional logging information
    }
