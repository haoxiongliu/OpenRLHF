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

from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code, DEFAULT_LAKE_PATH, DEFAULT_LEAN_WORKSPACE, DEFAULT_REPL_PATH, has_statement

logger = logging.getLogger("lean_reward_server")

class RewardRequest(BaseModel):
    queries: List[str]  # in fact prompt+response
    prompts: Optional[List[str]] = None  # in fact prompt only
    labels: Optional[List[str]] = None

class RewardConfig:
    def __init__(self, 
                 lake_path: str = None,
                 lean_workspace: str = None,
                 timeout: int = 300,
                 max_concurrent_requests: int = 16,
                 memory_limit: float = 5,
                 debug: bool = False,
                 use_pty: bool = False,
                 repl_path: str = None):
        self.lake_path = lake_path or DEFAULT_LAKE_PATH
        self.lean_workspace = lean_workspace or DEFAULT_LEAN_WORKSPACE
        self.repl_path = repl_path or DEFAULT_REPL_PATH
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.memory_limit = memory_limit
        self.debug = debug
        self.use_pty = use_pty
        logger.info(f"Initializing Lean4ServerScheduler with {max_concurrent_requests} concurrent requests and {memory_limit}GB memory limit")
        self.scheduler = Lean4ServerScheduler(
            max_concurrent_requests=self.max_concurrent_requests, 
            timeout=self.timeout, 
            memory_limit=self.memory_limit,
            name='reward_verifier',
            use_pty=self.use_pty,
            repl_path=self.repl_path
        )


def create_app(config: RewardConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Server starting up")
        yield
        if hasattr(app.state.config, 'scheduler'):
            app.state.config.scheduler.close()
            logger.info("Lean4ServerScheduler closed")
    
    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    
    def get_config(request: Request) -> RewardConfig:
        return request.app.state.config

    @app.post("/reward")
    async def get_reward(
        reward_request: RewardRequest,
        config: Annotated[RewardConfig, Depends(get_config)]
    ):
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[REQ-{request_id}] Received reward request with {len(reward_request.queries)} queries")

        # although reward does not to be 100% accurate
        # but loose rules can lead to reward hacking.
        codes = []
        if all([query.count("```lean4") == 1 for query in reward_request.queries]):
            mode = "completion"
        else:
            mode = "chat"
        for i in range(len(reward_request.queries)):
            query = reward_request.queries[i]
            if mode == "completion":
                code = extract_code(query, omit_think=True)
            else:
                # kimina prompt, need to extract the prefix from the prompt
                prompt = reward_request.prompts[i]
                response = query[len(prompt):]
                response_code = extract_code(response)
                prompt_code = extract_code(prompt)
                prefix = prompt_code.split(":=")[0]
                mc_prefix_end = response_code.find(":=")
                if mc_prefix_end == -1:
                    logger.debug(f"No := or code block found in ...{response[-50:]}")
                    code = "[[No := or code block found in the model output]]"
                else:
                    code = prefix + response_code[mc_prefix_end:]
            codes.append(code)
        verification_request_ids = config.scheduler.submit_all_request(codes)
        verification_results = await config.scheduler.async_get_all_request_outputs(verification_request_ids)
        rewards = [1.0 if result.get("complete", False) else 0.0 for result in verification_results]
        
        if config.debug:
            i = random.randint(0, len(reward_request.queries) - 1)
            debug_dict = {
                # "query": reward_request.queries[i],
                "code": codes[i],
                "reward": rewards[i],
                "errors": verification_results[i].get("errors", []),
            }
            logger.debug(f"\n[REQ-{request_id}] {debug_dict}")

        average_reward = sum(rewards) / len(rewards)
        logger.info(f"[REQ-{request_id}] Completed - Average reward: {average_reward}")

        return {
            "rewards": rewards,
        }
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean verification reward model API server")
    parser.add_argument("--host", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--lake_path", type=str, default=None, help="Lake executable path")
    parser.add_argument("--repl_path", type=str, default=None, help="Repl executable path")
    parser.add_argument("--lean_workspace", type=str, default=None, help="Lean workspace path")
    parser.add_argument("--timeout", type=int, default=60, help="DO NOT USE THIS SCRIPT FOR EVALUATION. Low timeout to encourage fast verification.")
    parser.add_argument("-n", "--max_concurrent", type=int, default=32, help="Maximum concurrent verification requests")
    parser.add_argument("--memory_limit", type=float, default=10, help="Memory limit in GB for Lean processes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--use_log_file", action="store_true", help="Use log file")
    parser.add_argument("--use_pty", action="store_true", default=True, help="Use pty mode")
    parser.add_argument("--no_use_pty", action="store_false", dest="use_pty")
    parser.add_argument("--pty_restart_count", type=int, default=100, help="Pty restart count")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
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
    
    config_instance = RewardConfig(
        lake_path=args.lake_path,
        lean_workspace=args.lean_workspace,
        timeout=args.timeout,
        max_concurrent_requests=args.max_concurrent,
        memory_limit=args.memory_limit,
        debug=args.debug,
        use_pty=args.use_pty,
        repl_path=args.repl_path
    )   
    
    app = create_app(config_instance)
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 