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
                    "description": "List of hammer tactics to use (e.g., ['simp', 'nlinarith', 'norm_cast'])",
                    "default": None
                },
                "hammer_recipe": {
                    "type": "string",
                    "description": "Hammer recipe/strategy to apply. Recommended choices: 'mix2' or 'mix4'",
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
    rewards: list[float | None]
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
        async with session.post(self.remote_rm_url, json=data, headers=headers, 
                              timeout=aiohttp.ClientTimeout(total=remote_timeout)) as response:
            response.raise_for_status()
            result = await response.json()
            result = RewardResponse(**result)
    return result

def _proofaug(self, args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ProofAug to enhance a proof with hammer tactics"""
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
    print(f"   📋 Hammer list: {hammer_list}")
    print(f"   🎯 Hammer recipe: {hammer_recipe}")
    print(f"   ⏱️  Step timeout: {step_timeout}s")
    print(f"   ⏰ Total timeout: {total_timeout}s")
    
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
            # Construct the query (original proof + statement)
            query = f"```lean4\n{node.proof_text}\n```"
            prompt = f"```lean4\n{node.statement}"
            label = node_id
            
            ret_obj = loop.run_until_complete(
                self._call_remote_reward_model(
                    queries=query,
                    prompts=prompt, 
                    labels=label,
                    proofaug_config=proofaug_config
                )
            )
        finally:
            loop.close()
        
        # Extract results
        reward = ret_obj.rewards[0]
        proofaug_code = ret_obj.proofaug_codes[0]
        proofaug_subst = ret_obj.proofaug_substs[0] 
        success_type = ret_obj.success_types[0]
        verify_time = ret_obj.verify_times[0]
        errors = ret_obj.errorss[0]
        
        print(f"   📊 ProofAug result: reward={reward}, success_type={success_type}")
        print(f"   ⏱️  Verification time: {verify_time}s")
        
        if success_type == "original":
            result = "success"
            message = "The original proof is already valid."
        elif success_type == "proofaug":
            result = "success"
            message = "ProofAug enhancement successful!"
            if proofaug_code:
                node.proof_text = proofaug_code
                node.status = "proved"
                print(f"   🔄 Updated proof with ProofAug enhancements")
        elif success_type == "pa_faield":
            result = "error"
            message = "ProofAug failed to correct the proof."

        return {
            "status": "completed",
            "result": result,
            "message": message,
            "success_type": success_type,
            "proofaug_subst": proofaug_subst,
            "proofaug_code": proofaug_code,
            "verify_time": verify_time,
            "errors": errors
        }
            
    except asyncio.TimeoutError:
        print(f"   ⏰ ProofAug timed out after {remote_timeout}s")
        return {
            "status": "error",
            "result": "timeout", 
            "message": f"ProofAug timed out after {remote_timeout} seconds",
            "suggestion": "Try increasing remote_timeout or total_timeout or use other meethods"
        }
    except Exception as e:
        print(f"   ❌ ProofAug error: {e}")
        return {
            "status": "error",
            "message": f"ProofAug failed with error: {str(e)}",
            "suggestion": "Check network connection and remote reward model server"
        }

# usage example
"""
Example tool call:
{
    "node_id": "root",
    "hammer_list": ["simp", "nlinarith", "norm_cast"],
    "hammer_recipe": "mix4",
    "step_timeout": 60,
    "total_timeout": 300,
    "remote_timeout": 300
}
Expected return:
{
    "status": "completed",
    "result": "success",
    "success_type": "proofaug",
    "proofaug_subst": {"0:5": "theorem example : 1 + 1 = 2 := by simp"},
    "proofaug_code": "theorem example : 1 + 1 = 2 := by simp",
    "verify_time": 1.2,
    "errors": [],
    "message": "ProofAug enhancement successful!"
}
""" 