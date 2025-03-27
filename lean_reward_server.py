import os
import re
import json
import torch
import logging
import argparse
import datetime
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from contextlib import asynccontextmanager
import random

from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"lean_reward_server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
                 debug: bool = False):
        self.lake_path = lake_path or os.path.expanduser('~/.elan/bin/lake')
        self.lean_workspace = lean_workspace or 'mathlib4/'
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.memory_limit = memory_limit
        self.debug = debug
        
        logger.info(f"Initializing Lean4ServerScheduler with {max_concurrent_requests} concurrent requests and {memory_limit}GB memory limit")
        self.scheduler = Lean4ServerScheduler(
            max_concurrent_requests=self.max_concurrent_requests, 
            timeout=self.timeout, 
            memory_limit=self.memory_limit,
            name='reward_verifier'
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
    
    @app.post("/reward")
    async def get_reward(reward_request: RewardRequest, request: Request):
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[REQ-{request_id}] Received reward request with {len(reward_request.queries)} queries")

        codes = [extract_code(query) for query in reward_request.queries]
        conf = request.app.state.config
        verification_request_ids = conf.scheduler.submit_all_request(codes)
        verification_results = await conf.scheduler.async_get_all_request_outputs(verification_request_ids)
        rewards = [1.0 if result.get("complete", False) else 0.0 for result in verification_results]

        if conf.debug:
            i = random.randint(0, len(reward_request.queries) - 1)
            debug_dict = {
                "query": reward_request.queries[i],
                "reward": rewards[i],
                "errors": verification_results[i].get("errors", []),
            }
            logger.debug(f"\n[REQ-{request_id}] {debug_dict}")

        average_reward = sum(rewards) / len(rewards)
        logger.info(f"[REQ-{request_id}] Completed - Average reward: {average_reward}")

        return {"rewards": rewards}
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean verification reward model API server")
    parser.add_argument("--host", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--lake_path", type=str, default=None, help="Lake executable path")
    parser.add_argument("--lean_workspace", type=str, default=None, help="Lean workspace path")
    parser.add_argument("--timeout", type=int, default=120, help="Verification timeout (seconds)")
    parser.add_argument("--max_concurrent", "-n", type=int, default=32, help="Maximum concurrent verification requests")
    parser.add_argument("--memory_limit", type=float, default=10, help="Memory limit in GB for Lean processes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--use_log_file", action="store_true", help="Use log file")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
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
        debug=args.debug
    )
    
    app = create_app(config_instance)
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 