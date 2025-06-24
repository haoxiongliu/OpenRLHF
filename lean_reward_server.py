import os
from os.path import join
import re
import json
import torch
import logging
import argparse
import datetime
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Annotated
import uvicorn
from contextlib import asynccontextmanager
import random
import signal
import sys


from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code, DEFAULT_LAKE_PATH, DEFAULT_LEAN_WORKSPACE, DEFAULT_REPL_PATH, has_statement, DEF_SIGN

logger = logging.getLogger("lean_reward_server")

# you should always modify this first before modifying the eval_pipeline.py and vllm_engine_async.py
class RewardRequest(BaseModel):
    queries: List[str]  # in fact prompt+response
    prompts: Optional[List[str]] = None  # in fact prompt only
    labels: Optional[List[str]] = None
    proofaug: bool = False
    hammer_list: Optional[List[str]|str] = None
    step_timeout: Optional[int] = None
    require_reconstruct: bool = False
    pa_with_orig: bool = False
        
def create_app(args: argparse.Namespace) -> FastAPI:
    # Initialize scheduler here instead of in Config class
    lake_path = args.lake_path or DEFAULT_LAKE_PATH
    lean_workspace = args.lean_workspace or DEFAULT_LEAN_WORKSPACE
    repl_path = args.repl_path or DEFAULT_REPL_PATH
    
    logger.info(f"Initializing Lean4ServerScheduler with {args.max_concurrent} concurrent requests and {args.memory_limit}GB memory limit")
    scheduler = Lean4ServerScheduler(
        max_concurrent_requests=args.max_concurrent, 
        timeout=args.step_timeout, 
        memory_limit=args.memory_limit,
        name='reward_verifier',
        use_pty=args.use_pty,
        repl_path=repl_path,
        lean_workspace=lean_workspace,
        lake_path=lake_path,
        pty_restart_count=args.pty_restart_count,
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Server starting up")
        yield
        if hasattr(app.state, 'scheduler'):
            app.state.scheduler.close()
            logger.info("Lean4ServerScheduler closed")
    
    app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.state.scheduler = scheduler
    
    def get_args(request: Request) -> argparse.Namespace:
        return request.app.state.args
    
    def get_scheduler(request: Request) -> Lean4ServerScheduler:
        return request.app.state.scheduler

    @app.post("/reward")
    async def get_reward(
        reward_request: RewardRequest,
        args: Annotated[argparse.Namespace, Depends(get_args)],
        scheduler: Annotated[Lean4ServerScheduler, Depends(get_scheduler)]
    ):
        """code MUST be included in a ```lean4 block, and we will extract the code."""
        n = len(reward_request.queries)
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[REQ-{request_id}] Received reward request with {len(reward_request.queries)} queries")

        # although reward does not to be 100% accurate
        # but loose rules can lead to reward hacking.
        codes = []
        mode = "completion"
        for query in reward_request.queries:
            if query.count("```lean4") > 1 or "<think>" in query or "<im_end>" in query:
                mode = "chat"
                break

        # codes_in_prompt = []
        for i in range(n):
            query = reward_request.queries[i]
            if not query:
                code = None
            elif mode == "completion":
                code = extract_code(query)
            elif mode == "chat":
                # kimina prompt, need to extract the prefix from the prompt
                prompt = reward_request.prompts[i]
                code_in_prompt = extract_code(prompt)
                response = query[len(prompt):]
                code_in_response = extract_code(response, omit_think=True)
                if code_in_response is None or code_in_prompt is None:
                    code = None
                else:
                    prefix = code_in_prompt.split(DEF_SIGN)[0]
                    sep_pos = code_in_response.find(DEF_SIGN)
                    if sep_pos == -1:
                        logger.debug(f"No {DEF_SIGN=} found in {code_in_response[-100:]}")
                        code = None
                    else:
                        code = prefix + code_in_response[sep_pos:]
            codes.append(code)
        tasks = [{
            "code": code,
            "proofaug": reward_request.proofaug,
            "hammer_list": reward_request.hammer_list,
            "require_reconstruct": reward_request.require_reconstruct,
            "step_timeout": reward_request.step_timeout,
            "pa_with_orig": reward_request.pa_with_orig,
        } for code in codes]
        
        verification_request_ids = scheduler.submit_all_request(tasks)
        verification_results = await scheduler.async_get_all_request_outputs(verification_request_ids)
        # The result is _verify_lean4_with_persistent_repl return value

        verify_times = [result.get("verify_time", 0.0) for result in verification_results]
        proofaug_bodies = [result.get("proofaug_body", None) for result in verification_results]
        success_types = [result.get("success_type", None) for result in verification_results]
        errorss = [result.get("errors", None) for result in verification_results]

        rewards = []
        for i in range(n):
            if verification_results[i].get("complete", False):
                reward = 1.0 - args.time_reward_ratio * min(verify_times[i]/args.time_reward_threshold, 1.0)
            else:
                reward = 0.0
            rewards.append(reward)

        i = random.randint(0, n - 1)
        debug_dict = {
            # "query": reward_request.queries[i],
            "code": codes[i],
            "reward": rewards[i],
            "proofaug_body": proofaug_bodies[i],
            "success_type": success_types[i],
            "errors": errorss[i],
            "verify_time": verify_times[i],
        }
        logger.debug(f"\n[REQ-{request_id}] {debug_dict}")

        average_reward = sum(rewards) / len(rewards)
        logger.info(f"[REQ-{request_id}] Completed - Average reward: {average_reward}")

        proofaug_codes = [] # prompt prefix + proofaug proof after sep
        for i, proofaug_body in enumerate(proofaug_bodies):
            if proofaug_body is None:
                proofaug_codes.append(None)
            else:
                assert isinstance(proofaug_body, str), f"Proofaug body is not a string: {proofaug_body}"
                code = codes[i] # type: str
                sep_pos = code.find(DEF_SIGN)
                proofaug_proof = proofaug_body.partition(DEF_SIGN)[2]
                proofaug_codes.append(code[:sep_pos] + DEF_SIGN + proofaug_proof)

        return {
            "rewards": rewards,
            "proofaug_codes": proofaug_codes,
            "success_types": success_types,
            "verify_times": verify_times,
            "errorss": errorss,
        }
    
    return app

def signal_handler(sig, frame):
    logger.info("Received interrupt signal, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Lean verification reward model API server")
    parser.add_argument("--host", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--lake_path", type=str, default=None, help="Lake executable path")
    parser.add_argument("--repl_path", type=str, default=None, help="Repl executable path")
    parser.add_argument("--lean_workspace", type=str, default=None, help="Lean workspace path")
    parser.add_argument("-n", "--max_concurrent", type=int, default=32, help="Maximum concurrent verification requests")
    parser.add_argument("--memory_limit", type=float, default=10, help="Memory limit in GB for Lean processes")
    parser.add_argument("--log_level", type=str, default="info", help="debug, info, warning, error, critical")
    parser.add_argument("--use_log_file", action="store_true", help="Use log file")
    parser.add_argument("--use_pty", action="store_true", default=True, help="Use pty mode")
    parser.add_argument("--no_use_pty", action="store_false", dest="use_pty")
    parser.add_argument("--pty_restart_count", type=int, default=10, help="Pty restart count")
    parser.add_argument("--step_timeout", type=int, default=60, help="Step timeout for the lean server")
    parser.add_argument("--time_reward_threshold", type=int, default=120, help="Time reward threshold in seconds")
    parser.add_argument("--time_reward_ratio", type=float, default=0.0, help="Use elapsed time as reward (not implemented yet)")
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"lean_reward_server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ] if args.use_log_file else [logging.StreamHandler()]
    )
    
    logger.info(f"Starting server with config: {vars(args)}")
    
    app = create_app(args)
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 