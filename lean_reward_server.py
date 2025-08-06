import logging
import argparse
import uuid

from fastapi import FastAPI, Request, Depends
from typing import Annotated
import uvicorn
from contextlib import asynccontextmanager
import random
import signal
import sys
import yaml
from os.path import join


from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code, PROJ_DIR, DEFAULT_LAKE_PATH, DEFAULT_LEAN_WORKSPACE, DEFAULT_REPL_PATH, has_statement, DEF_SIGN, split_header_body
from prover.agent_utils import RewardResponse, RewardRequest
from prover.logger import logger, set_log_level


def create_app(args: argparse.Namespace) -> FastAPI:
    # Initialize scheduler here instead of in Config class
    with open(join(PROJ_DIR, "configs", "lean_env", f"{args.config_name}.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    lake_path = cfg.get("lake_path", args.lake_path or DEFAULT_LAKE_PATH)
    if cfg.get("lean_workspace"):
        if cfg.get("use_relative_path", True):
            lean_workspace = join(PROJ_DIR, cfg.get("lean_workspace"))
        else:
            lean_workspace = cfg.get("lean_workspace")
    else:
        lean_workspace = args.lean_workspace or DEFAULT_LEAN_WORKSPACE
    if cfg.get("repl_path"):
        if cfg.get("use_relative_path", True):
            repl_path = join(PROJ_DIR, cfg.get("repl_path"))
        else:
            repl_path = cfg.get("repl_path")
    else:
        repl_path = args.repl_path or DEFAULT_REPL_PATH
    
    logger.info(f"Using {lake_path=} {repl_path=} {lean_workspace=}")
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

    @app.post("/reward", response_model=RewardResponse)
    async def get_reward(
        reward_request: RewardRequest,
        args: Annotated[argparse.Namespace, Depends(get_args)],
        scheduler: Annotated[Lean4ServerScheduler, Depends(get_scheduler)]
    ):
        """code MUST be included in a ```lean4 block, and we will extract the code."""
        n = len(reward_request.queries)
        request_id = str(uuid.uuid4())[:8]
        logger.debug(f"[REQ-{request_id}] Received reward request with {len(reward_request.queries)} queries")

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
                if not prompt:
                    logger.warning(f"No prompt found for chat mode {query=}.")
                code_in_prompt = extract_code(prompt)
                response = query[len(prompt):]
                code_in_response = extract_code(response, omit_think=True)
                # TODO: use pattern match "theorem .*DEF_SIGN" to find the statement
                # The original implement fails when the informal includes, and for lemma styles + defs.
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
        headers = [split_header_body(code, remove_comments=False)[0] if code is not None else "" for code in codes]
        tasks = [{
            "code": code,
            "proofaug": reward_request.proofaug,
            "record_pa_reward": reward_request.record_pa_reward,
            "hammer_list": reward_request.hammer_list,
            "hammer_recipe": reward_request.hammer_recipe,
            "random_order": reward_request.random_order,
            "require_reconstruct": reward_request.require_reconstruct,
            "step_timeout": reward_request.step_timeout,
            "total_timeout": reward_request.total_timeout,
            "pa_with_orig": reward_request.pa_with_orig,
            "non_repl": reward_request.non_repl,
        } for code in codes]
        
        verification_request_ids = scheduler.submit_all_request(tasks)
        verification_results: list[dict] = await scheduler.async_get_all_request_outputs(verification_request_ids)
        # The result is _verify_lean4_with_persistent_repl return value
        verify_times = [result.get("verify_time", None) for result in verification_results]
        search_times = [result.get("search_time", None) for result in verification_results]
        proofaug_bodies = [result.get("proofaug_body", None) for result in verification_results]
        bodies = [result.get("body", None) for result in verification_results]
        success_types = [result.get("success_type", None) for result in verification_results]
        errorss = [result.get("errors", None) for result in verification_results]
        proofaug_substs = [result.get("proofaug_subst", None) for result in verification_results]
        pa_depths = [result.get("pa_depth", 0) for result in verification_results]
        depths = [result.get("depth", 0) for result in verification_results]

        rewards = []
        orig_rewards = []
        pa_rewards = []
        for i in range(n):
            success_type = success_types[i]
            orig_reward = 1.0 if success_type in ["pa_orig", "original"] else 0.0
            pa_reward = 1.0 if success_type in ["pa_orig", "original", "proofaug"] else 0.0
            reward = pa_reward if reward_request.proofaug else orig_reward
            if orig_reward != pa_reward:
                logger.info(f"proofaug reward modification detected:\n{proofaug_bodies[i]=}\nfrom\n{bodies[i]=}")

            rewards.append(reward)
            orig_rewards.append(orig_reward)
            pa_rewards.append(pa_reward)

        i = random.randint(0, n - 1)
        average_reward = sum(rewards) / len(rewards)
        logger.debug(f"[REQ-{request_id}] Completed - Average reward: {average_reward}")

        proofaug_codes = [] # prompt prefix + proofaug proof after sep
        for i, proofaug_body in enumerate(proofaug_bodies):
            if proofaug_body is None:
                proofaug_codes.append(None)
            else:
                assert isinstance(proofaug_body, str)
                code: str = codes[i]
                sep_pos = code.find(DEF_SIGN)
                proofaug_proof = proofaug_body.partition(DEF_SIGN)[2] # this is correct
                proofaug_codes.append(code[:sep_pos] + DEF_SIGN + proofaug_proof)

        response = RewardResponse(
            rewards=rewards,
            orig_rewards=orig_rewards,
            pa_rewards=pa_rewards,
            bodies=bodies,
            headers=headers,
            proofaug_substs=proofaug_substs,
            proofaug_codes=proofaug_codes,
            success_types=success_types,
            verify_times=verify_times,
            search_times=search_times,
            errorss=errorss,
            pa_depths=pa_depths,
            depths=depths,
        )
        logger.debug(f"\n[REQ-{request_id}] {response}")

        return response
    
    return app

def signal_handler(sig, frame):
    # logger.info("Received interrupt signal, shutting down gracefully...")
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
    parser.add_argument("--config_name", type=str, default="default", help="Lean environment config name, see configs/lean_env/")
    parser.add_argument("-n", "--max_concurrent", type=int, default=32, help="Maximum concurrent verification requests")
    parser.add_argument("--memory_limit", type=float, default=10, help="Memory limit in GB for Lean processes")
    parser.add_argument("--log_level", type=str, default="info", help="debug, info, warning, error, critical")

    parser.add_argument("--use_pty", action="store_true", default=True, help="Use pty mode")
    parser.add_argument("--no_use_pty", action="store_false", dest="use_pty")
    parser.add_argument("--pty_restart_count", type=int, default=100, help="Pty restart count")
    parser.add_argument("--step_timeout", type=int, default=180, help="default step timeout for the lean server")
    args = parser.parse_args()
    
    # Set log level through prover.logger
    set_log_level(args.log_level.upper())
    
    logger.info(f"Starting server with config: {vars(args)}")
    
    app = create_app(args)
    logger.info(f"Starting server on {args.host}:{args.port}")
    if args.log_level.lower() in ["warning"]:
        logger.warning("Warning log level is set, most logs will be suppressed.")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower()) 