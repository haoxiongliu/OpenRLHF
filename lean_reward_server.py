import os
import re
import json
import torch
import logging
import argparse
import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
from contextlib import asynccontextmanager

from prover.lean.verifier import Lean4ServerScheduler

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"lean_reward_server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("lean_reward_server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Server starting up")
    yield
    # Shutdown code
    if hasattr(config, 'scheduler'):
        config.scheduler.close()
        logger.info("Lean4ServerScheduler closed")

app = FastAPI(lifespan=lifespan)

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
                 debug: bool = False):
        self.lake_path = lake_path or os.path.expanduser('~/.elan/bin/lake')
        self.lean_workspace = lean_workspace or 'mathlib4/'
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.debug = debug
        
        # Initialize Lean4ServerScheduler
        logger.info(f"Initializing Lean4ServerScheduler with {max_concurrent_requests} concurrent requests")
        self.scheduler = Lean4ServerScheduler(
            max_concurrent_requests=self.max_concurrent_requests, 
            timeout=self.timeout, 
            name='reward_verifier'
        )

config = RewardConfig()

def extract_code(text: str) -> Optional[str]:
    code = None
    if m:= re.search(r'```lean4\n(.*?)\n```', text, re.DOTALL):
        code = m.group(1)   
    elif m:= re.search(r'```\S*\n(.*)', text, re.DOTALL):
        code = m.group(1)   # no ``` case
    else:
        code = "[[No code found.]]"
    return code

@app.post("/reward")
async def get_reward(request: RewardRequest):
    """
    Calculate rewards for Lean code
    
    Args:
        queries: Required query list (prompt+response)
        prompts: Optional prompt list
        labels: Optional label list
    
    Returns:
        Dict with "rewards" key containing the reward values
    """
    logger.info(f"Received reward request with {len(request.queries)} queries")
    
    # Extract and verify code
    codes = [extract_code(query) for query in request.queries]
    request_id_list = config.scheduler.submit_all_request(codes)
    verification_results = config.scheduler.get_all_request_outputs(request_id_list)
    rewards = [1.0 if result["complete"] else 0.0 for result in verification_results]
    
    # if config.debug:
    for i in range(len(request.queries)):
        log_dict = {
            "query": request.queries[i],
            "reward": rewards[i],
            # "code": codes[i],
            # "errors": verification_results[i].get("errors", []),
        }    
        if config.debug:
            log_dict["errors"] = verification_results[i].get("errors", [])
        logger.debug(f"\n{log_dict}")
    
    # Return in format expected by OpenRLHF
    return {"rewards": rewards}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean verification reward model API server")
    parser.add_argument("--host", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--lake_path", type=str, default=None, help="Lake executable path")
    parser.add_argument("--lean_workspace", type=str, default=None, help="Lean workspace path")
    parser.add_argument("--timeout", type=int, default=300, help="Verification timeout (seconds)")
    parser.add_argument("--max_concurrent", type=int, default=16, help="Maximum concurrent verification requests")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    
    args = parser.parse_args()
    
    # Configure logging based on debug mode
    log_level = logging.DEBUG if args.debug else logging.INFO
    # only add filehandler if debug is True
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ] if args.debug else [logging.StreamHandler()]
    )
    
    logger.info(f"Starting server with config: {vars(args)}")
    
    # Update configuration
    config = RewardConfig(
        lake_path=args.lake_path,
        lean_workspace=args.lean_workspace,
        timeout=args.timeout,
        max_concurrent_requests=args.max_concurrent,
        debug=args.debug
    )
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 