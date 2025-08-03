"""
ProofAug Tool Integration Snippet

Essential code snippets to add ProofAug tool to your existing pure agentic prover.
Copy these parts into your agent implementation following the integration guide.
"""

# ============================================================================
# 1. ADD TO self.tools LIST (around line 76 in your agent)
# ============================================================================

PROOFAUG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proofaug",
        "description": "Apply ProofAug proof augmentation to enhance a proof using hammer tactics and timeout optimization",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "ID of the theorem node to apply ProofAug to"
                },
                "hammer_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hammer tactics to use (e.g., ['simp', 'auto', 'blast'])",
                    "default": None
                },
                "hammer_recipe": {
                    "type": "string",
                    "description": "Hammer recipe/strategy to apply",
                    "default": None
                },
                "step_timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for individual proof steps",
                    "default": 60
                },
                "total_timeout": {
                    "type": "integer", 
                    "description": "Total timeout in seconds for the entire ProofAug process",
                    "default": None
                },
                "remote_timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the remote API call",
                    "default": 300
                }
            },
            "required": ["node_id"]
        }
    }
}

# ============================================================================
# 2. ADD TO _execute_function METHOD ROUTING
# ============================================================================

# Add this line to your _execute_function method:
# elif function_name == "proofaug":
#     return self._proofaug(arguments)

# ============================================================================
# 3. IMPLEMENTATION METHOD - ADD TO YOUR AGENT CLASS
# ============================================================================

import asyncio
import aiohttp
from typing import Any, Dict
from prover.agent_utils import RewardResponse, RewardRequest
from pydantic import BaseModel

class RewardRequest(BaseModel):
    """
    This class is used to send request to lean_reward_server.
    """
    queries: list[str]  # in fact prompt+response
    prompts: list[str] | None = None  # in fact prompt only
    labels: list[str] | None = None
    proofaug: bool = False
    record_pa_reward: bool = False
    hammer_list: list[str|None] | str | None = None
    hammer_recipe: str | None = None
    random_order: bool = False  # random execution order for hammers when applying proofaug
    step_timeout: float | None = None
    total_timeout: float | None = None
    require_reconstruct: bool = False
    pa_with_orig: bool = False
    non_repl: bool = False
    time_reward_ratio: float = 0.0
    time_reward_threshold: float = 120.0
    depth_reward_ratio: float = 0.0
    depth_reward_rate: float = 0.25


class RewardResponse(BaseModel):
    """
    when RewardResponse(**dict) receive extra fields, it will be ignored.
    """
    proofaug_substs: list[dict | None]
    proofaug_codes: list[str | None]
    success_types: list[str | None]
    verify_times: list[float | None]
    errorss: list[list[str]]


REMOTE_RM_URL = "http://localhost:5000/reward"

async def _call_remote_reward_model(self, queries, prompts, labels, **kwargs) -> RewardResponse:
    """Call remote reward model for ProofAug"""
    proofaug_config = kwargs.get("proofaug_config", {})
    hammer_list = proofaug_config.get("hammer_list", None)
    hammer_recipe = proofaug_config.get("hammer_recipe", None)
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
        proofaug=True,
        hammer_list=hammer_list,
        hammer_recipe=hammer_recipe,
        require_reconstruct=True,
        step_timeout=step_timeout,
        pa_with_orig=True,
        total_timeout=total_timeout,
    ).model_dump(exclude_none=True)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(REMOTE_RM_URL, json=data, headers=headers, 
                              timeout=aiohttp.ClientTimeout(total=remote_timeout)) as response:
            response.raise_for_status()
            result = await response.json()
            result = RewardResponse(**result)
    return result

def _proofaug(self, args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ProofAug to enhance a proof with hammer tactics"""
    remote_rm_url = args.get("remote_rm_url", REMOTE_RM_URL)
    requested_node_id = args["node_id"]
    hammer_list = args.get("hammer_list", None)
    hammer_recipe = args.get("hammer_recipe", None)
    step_timeout = args.get("step_timeout", 60)
    total_timeout = args.get("total_timeout", None)
    remote_timeout = args.get("remote_timeout", 300)
    
    # Resolve node ID with fallback logic
    node_id = self._resolve_node_id(requested_node_id)
    if node_id is None:
        return {"status": "error", "message": f"Node {requested_node_id} not found"}
    
    node = self.nodes[node_id]
    
    if not node.proof_text:
        return {
            "status": "error", 
            "message": f"Node {node_id} has no proof text to apply ProofAug to",
            "suggestion": f"Try prove_directly(node_id='{node_id}') first to generate a proof"
        }
    
    print(f"🔧 APPLYING PROOFAUG: {node_id}")
    self.strategies_used.append("proofaug")
    
    try:
        # Prepare ProofAug configuration
        proofaug_config = {
            "proofaug": True,
            "hammer_list": hammer_list,
            "hammer_recipe": hammer_recipe,
            "step_timeout": step_timeout,
            "total_timeout": total_timeout,
            "remote_timeout": remote_timeout
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ret_obj = loop.run_until_complete(
                self._call_remote_reward_model(
                    queries=node.proof_text,
                    prompts=node.statement, 
                    labels=node_id,
                    proofaug_config=proofaug_config
                )
            )
        finally:
            loop.close()
        
        # Extract results
        proofaug_code = ret_obj.proofaug_codes[0]
        proofaug_subst = ret_obj.proofaug_subst[0] 
        success_type = ret_obj.success_types[0]
        verify_time = ret_obj.verify_times[0]
        
        if success_type == "proofaug":
            # ProofAug succeeded - update node
            if proofaug_code:
                node.proof_text = proofaug_code
                node.status = "proved"
            
            return {
                "status": "completed",
                "result": "enhanced",
                "reward": reward,
                "success_type": success_type, # original, failed, proofaug, pa_failed
                "proofaug_subst": proofaug_subst, # dict[line_number_range: proofaug_code_snippet]
                "proofaug_code": proofaug_code, # str
                "verify_time": verify_time,
                "number_of_substitutions_applied": len(proofaug_subst) if proofaug_subst else 0,
                "message": f"ProofAug successfully find a proof",
            }
        else:
            return {
                "status": "completed", 
                "result": "no_improvement",
                "reward": reward,
                "proofaug_subst": proofaug_subst,
                "message": f"ProofAug completed but did not improve the proof",
                "suggestion": "Try different hammer tactics or adjust timeouts"
            }
            
    except asyncio.TimeoutError:
        return {
            "status": "completed",
            "result": "timeout", 
            "message": f"ProofAug timed out after {remote_timeout} seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "result": None,
            "message": f"ProofAug failed with error: {str(e)}"
        }

# usage example
"""
Example tool call:
{
    "node_id": "root",
    "hammer_list": ["simp", "auto", "blast"],
    "hammer_recipe": "standard",
    "step_timeout": 60,
    "total_timeout": 300,
    "remote_timeout": 300
}
Expected return:
{
    "status": "completed",
    "result": "enhanced",
    "proofaug_subst": {"0:5": "theorem example : 1 + 1 = 2 := by simp"},
    "proofaug_code": "theorem example : 1 + 1 = 2 := by simp",
    "message": "ProofAug successfully enhanced proof"
}
""" 