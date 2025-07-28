# Lean4 verifier implementation based on DeepSeek-Prover-V1.5
import os
import time
import json
import ctypes
import traceback
import subprocess
import multiprocessing as mp
import select
import pty
import termios
import signal
import resource
import itertools
import concurrent.futures
from typing import Optional
from collections import deque

from prover.lean.ast_parser import lean4_parser
from prover.workers import ProcessScheduler
from prover.logger import logger
from prover.constants import HINT_DICT, RECIPE2HAMMER_LIST
from prover.utils import DEFAULT_LAKE_PATH, DEFAULT_LEAN_WORKSPACE, DEFAULT_REPL_PATH, split_header_body, to_command, n_indent, compile_errors, is_complete
from prover.lean.psa import ProposalStructure, Snippet, Block, BlockState


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, timeout=300, memory_limit=-1, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, repl_path=DEFAULT_REPL_PATH, use_pty=False, pty_restart_count=100):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
        self.lake_path = lake_path
        self.repl_path = repl_path
        self.lean_workspace = lean_workspace
        self.header_dict = {}  # Dictionary to store different headers and their environments
        self.use_pty = use_pty
        self.pty_restart_count = pty_restart_count
        self.restart_count = 0
        self.repl_process = None
        self.latest_state = None
        self.recent_commands = deque(maxlen=100) # max size 100
        
    def _clean_init_repl(self):
        """Create a REPL process using a pseudo-terminal"""
        
        self._cleanup_repl()
        
        try:
            self.master_fd, slave_fd = pty.openpty()
            self._set_raw_mode(self.master_fd)

            # Define memory limit setup
            def set_mem_limit():
                if self.memory_limit > 0:
                    soft_limit = int(self.memory_limit * 1024 ** 3)  # Convert GB to bytes
                    hard_limit = int(soft_limit + 1024 ** 3)  # 1GB extra
                    resource.setrlimit(
                        resource.RLIMIT_AS, 
                        (soft_limit, hard_limit)  # (soft, hard)
                    )
            # legacy [self.lake_path, "exe", 'repl']
            self.repl_process = subprocess.Popen(
                [self.lake_path, "env", self.repl_path],
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=subprocess.DEVNULL,
                text=False,
                cwd=self.lean_workspace,
                start_new_session=True,
                preexec_fn=set_mem_limit if self.memory_limit > 0 else None
            )
            # Close slave fd since master process will use master_fd
            os.close(slave_fd)
            return True
        except Exception as e:
            print(f"Failed to initialize REPL: {str(e)}")
            logger.error(f"Process {self.idx}: Failed to initialize REPL: {str(e)}", stack_info=True)
            return False
    
    def _reset_latest_state(self):
        self.latest_state = None

    def _to_command(self,code, env=None, proofState=None, sorries=None, verbose=False, use_latest_state=False):
        if env != None:
            state = {"env": env}
        elif proofState != None:
            state = {"proofState": proofState}
        elif use_latest_state:
            env_keys = ["env", "proofState"]
            state = {k: v for k, v in self.latest_state.items() if k in env_keys} if self.latest_state else {}
        else:
            state = {}
        code_key = "tactic" if "proofState" in state.keys() else "cmd"
        cmd = {code_key: code}
        cmd.update(state)
        if sorries is not None: # "grouped" or "individual"
            cmd["sorries"] = sorries
        if verbose:
            print(json.dumps(cmd, ensure_ascii=False))
        return cmd

    def repl_run(self, code, env=None, proofState=None, sorries=None, verbose=False):
        cmd = self._to_command(code, env, proofState, sorries, verbose)
        return self._send_command_to_repl(cmd)

    def _set_raw_mode(self, fd):
        """Set terminal to raw mode for better process interaction"""
        try:
            attrs = termios.tcgetattr(fd)
            attrs[3] = attrs[3] & ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception as e:
            logger.error(f"Warning: Failed to set raw mode: {str(e)}")
            raise e
    
    def _initialize_header_env(self, header):
        """Initialize the environment for a given header"""
        if header in self.header_dict:
            return self.header_dict[header]
        
        self._clean_init_repl()
        result = self._send_command_to_repl(to_command(header, env=None))
        if 'env' not in result:
            messages = result.get('messages', [])   
            logger.debug(f"Process {self.idx}: Failed to initialize {header=} with {messages=}", stack_info=False)
            return None
        else:
            self.header_dict[header] = result['env']
            return result['env']

    
    def _send_command_to_repl(self, command: dict, timeout: Optional[float] = None):
        """Send command to REPL with enhanced message framing"""
        ret_obj = None
        try:
            if timeout is None:
                timeout = self.timeout
            
            # load last command buffer
            while True:
                ready, _, _ = select.select([self.master_fd], [], [], 0.0)
                if not ready:
                    break
                last_chunk = os.read(self.master_fd, 4096)  # Discard any buffered data
                logger.warning(f"Non-read buffer data: {last_chunk}")
            
            # Send with protocol delimiter
            os.write(self.master_fd, (json.dumps(command) + "\n\n").encode())
            self.recent_commands.append(command)
            response = b''
            start_time = time.time()
            max_size = 10 * 1024 * 1024  # 10MB safety limit


            while True:
                remaining = max(0.1, timeout - (time.time() - start_time))
                ready, _, _ = select.select([self.master_fd], [], [], remaining)
                if not ready:
                    raise TimeoutError(f"error: REPL process timeout after {timeout} seconds")
                
                # Read chunk
                try:
                    chunk = os.read(self.master_fd, 4096)
                except (OSError, IOError) as e:
                    raise ValueError(f"Unexpected error reading from REPL: {str(e)}")             
                if not chunk:  # EOF
                    break
                    
                response += chunk
                if len(response) > max_size:
                    raise ValueError(f"REPL process response too large {len(response)=} > {max_size=}")

                # REPL protocol delimiter
                if b'\r\n\r\n' in response:
                    msg, _, tail = response.partition(b'\r\n\r\n')
                    if tail != b'':
                        raise ValueError(f"Process {self.idx}: Warning - REPL process response is not a valid JSON: {response}")
                    response = msg  # Keep remaining data
                    break

            if not response:
                raise ValueError("Empty response from REPL")
            
            try:
                response_obj = json.loads(response.decode())
                if 'proofState' in response_obj:
                    self.latest_state = {"proofState": response_obj['proofState']}
                elif 'sorries' in response_obj:
                    self.latest_state = response_obj['sorries'][0]
                elif 'env' in response_obj:
                    self.latest_state = {"env": response_obj['env']}

                ret_obj = response_obj
            except Exception as e:
                ret_obj = {"messages": [{"data": "response handling error: " + e.__class__.__name__ + str(e), "severity": "error"}]}
            
            ret_obj['command'] = command
            return ret_obj
        
        except TimeoutError as e:
            self._clean_init_repl()
            raise e

        except Exception as e:
            self._clean_init_repl()
            raise ValueError(f"{e.__class__.__name__} {e}, resulting in repl restarted, during sending {command=}")

    def _verify_lean4_with_persistent_repl(
        self, 
        code: Optional[str], 
        allTactics: bool=False, 
        ast: bool=False, 
        premises: bool=False, 
        tactics: bool=False, 
        proofaug: bool=False,
        pa_with_orig: bool=False,
        hammer_type: Optional[str]=None,    # legacy
        hammer_list: Optional[list[str] | str]=None,
        hammer_recipe: Optional[str]=None,
        require_reconstruct: bool=False,
        step_timeout: Optional[float]=None,
        sorry_mode: str='individual',   # 'individual' or 'grouped'
        non_repl: bool=False,
    ):
        if not code:
            return {
                "pass": False,
                "complete": False,
                "errors": ["No code found in the request"]
            }
        assert not non_repl or not proofaug, "non_repl and proofaug cannot be both True"

        global hammer_count
        start_time = time.time()
        hammer_count = 0  # Initialize hammer_count at the start
        complete = False
        try:
            if non_repl:
                raise NotImplementedError("non_repl is not implemented yet")
            
            header, body = split_header_body(code, remove_comments=True)    # this header logic is redundant?
            orig_body = body
            orig_header = header
            init_env = None

            if hammer_recipe:
                hammer_list = RECIPE2HAMMER_LIST[hammer_recipe]
            if not hammer_list:
                hammers = [HINT_DICT[hammer_type]] # can be None, for no hammer ablation. remember to check.
            else:
                hammers = [HINT_DICT[hammer_list]] if isinstance(hammer_list, str) else [HINT_DICT[ht] for ht in hammer_list]
            hammers = [h for h in hammers if h is not None]
            if any("hammer" in h for h in hammers):
                header = "import Hammer\n" + header
                init_env = self._initialize_header_env(header)
            elif header is not None:
                init_env = self._initialize_header_env(header)

            if (not proofaug) or pa_with_orig:

                command = dict(cmd=body, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
                if init_env is not None:
                    command.update(env=init_env)
                result = self._send_command_to_repl(command, timeout=step_timeout)
                
                complete = is_complete(result, code)
                verification_result = {
                    "success_type": 'original' if complete else 'failed',
                    "sorries": result.get('sorries', []), 
                    "tactics": result.get('tactics', []),
                    "errors": compile_errors(result),
                    "messages": result.get('messages', []),
                    "system_errors": [],
                    "ast": lean4_parser(code, result['ast']) if 'ast' in result else {},
                    "header": orig_header,
                    "body": orig_body,
                    "complete": complete,
                    # "verified_code": code,  # Keep original code for reference
                }

            if proofaug and not complete:
                assert self.use_pty, "ProofAug is only supported in Pty mode"
                body = body.replace("all_goals ", "")
                prop_struct = ProposalStructure(body)
                block = prop_struct.root.parts[0]
                proofaug_subst = dict()
                ps2goals = {None: []}

                def verify_block(block: Block, ps: Optional[int] = None) -> Optional[int]:
                    # handle statement individually. then the rest is handled by verify_block
                    global hammer_count
                    init_ps = ps
                    init_goals = ps2goals[ps]
                    sttm_part = block.parts[0]
                    assert sttm_part.category == 'statement' # this shoule be asserted by _analyze
                    code = sttm_part.content
                    # handle ps=None case in advance. level=0 block is special.
                    if ps is None:
                        assert block.level == 0, "ps=None should only happen in level=0 block"
                        if not code.strip().endswith('by'): # single tactic. handle just in case pa_with_orig is False.
                            cmd = to_command(code, env=init_env, sorries=sorry_mode)
                            result = self._send_command_to_repl(cmd, timeout=step_timeout)
                            if is_complete(result, code):
                                block.state = BlockState.COMPLETED
                            else: # we do not handle this case. just let it fail.
                                block.state = BlockState.STTM_FAILED
                            return init_ps, result
                        else:
                            code += ' sorry'
                            cmd = to_command(code, env=init_env, sorries=sorry_mode)
                            result = self._send_command_to_repl(cmd, timeout=step_timeout)
                            if compile_errors(result):
                                block.state = BlockState.STTM_FAILED
                                return init_ps, result
                            ps = result['sorries'][0]['proofState']
                            ps2goals[ps] = result['sorries'][0]['goals']

                    else:
                        if code.strip().endswith('by'):
                            code += ' sorry'
                        # elif "\n" not in code.split('by')[-1] use flag!
                        cmd = to_command(code, proofState=ps, sorries=sorry_mode)
                        result = self._send_command_to_repl(cmd, timeout=step_timeout)
                        if compile_errors(result):
                            block.state = BlockState.STTM_FAILED
                            return init_ps, result
                        ps = result['proofState']
                        ps2goals[ps] = result['goals']

                    sttm_ps = ps  # the sorry state. 
                    # handle the rest of the block. ps tracks the latest valid proofState.
                    rest_parts = block.parts[1:] if len(block.parts) > 1 else []
                    rest_part_index = 0
                    for part in rest_parts:
                        if isinstance(part, Snippet):
                            code = part.content
                            cmd = to_command(code, proofState=ps, sorries=sorry_mode)
                            result = self._send_command_to_repl(cmd, timeout=step_timeout)
                            if compile_errors(result):
                                break
                            ps = result['proofState']
                            ps2goals[ps] = result['goals']
                        elif isinstance(part, Block):   # equivalent to has_statement in this line
                            ps, result = verify_block(part, ps)
                            if part.state != BlockState.COMPLETED:
                                break
                        rest_part_index += 1
                        if len(ps2goals[ps]) == len(init_goals) and rest_part_index != len(rest_parts):
                            block._proofaug_parts = block.parts[:rest_part_index+1]
                            proofaug_subst[f"{block.start_line}:{block.end_line}"] = block.proofaug_content
                            break
                    
                    # check whether the current block is completed by checking the number of goals
                    assert len(ps2goals[ps]) >= len(init_goals), f"Observe {len(ps2goals[ps])=} < {len(init_goals)=} in {block=} at {part=}"
                    if len(ps2goals[ps]) > len(init_goals): # proofaug
                        ps_cands = sorted(set([ps, sttm_ps]), reverse=True)
                        cand_combs = list(itertools.product(ps_cands, hammers))
                        # removed the cand_i = len(cand_combs) - 1 case since cand_combs could be empty
                        for cand_i, (ps_cand, hammer) in enumerate(cand_combs):
                            cmd = to_command(hammer, proofState=ps_cand, sorries=sorry_mode)
                            result = self._send_command_to_repl(cmd, timeout=step_timeout)
                            hammer_count += 1
                            if compile_errors(result):
                                continue
                            ps_new = result['proofState']
                            ps2goals[ps_new] = result['goals']
                            if len(ps2goals[ps_new]) == len(init_goals):
                                ps = ps_new
                                sttm_indent = n_indent(sttm_part.content)
                                last_part = block.parts[rest_part_index]
                                last_indent = n_indent(last_part.content)
                                if "hammer" in hammer:
                                    hammer_message = result["messages"][0]["data"]
                                    hammer_output = hammer_message.split("Try this:\n")[1] if "hammer" in hammer else hammer
                                    assert hammer_output[0] != "\n"
                                    if ps_cand == sttm_ps:
                                        connect = "\n" if hammer_output.startswith(" ") else " "
                                        hammer_output = connect + hammer_output
                                        hammer_output = hammer_output.replace("\n", "\n" + " "*sttm_indent)
                                        sttm_snippet = Snippet(block.statement + ':= by' + hammer_output)
                                        block._proofaug_parts = [sttm_snippet]
                                    else:
                                        prefix = " "*sttm_indent if hammer_output.startswith(" ") else " "*last_indent
                                        hammer_output = hammer_output.replace("\n", "\n" + prefix)
                                        hammer_snippet = Snippet(prefix + hammer_output)
                                        block._proofaug_parts = block.parts[:rest_part_index+1] + [hammer_snippet]
                                else:
                                    if ps_cand == sttm_ps:
                                        sttm_snippet = Snippet(block.statement + ':= by ' + hammer)
                                        block._proofaug_parts = [sttm_snippet]
                                    else:
                                        hammer_snippet = Snippet(" "*last_indent + hammer)
                                        block._proofaug_parts = block.parts[:rest_part_index+1] + [hammer_snippet]
                                block.state = BlockState.COMPLETED
                                proofaug_subst[f"{block.start_line}:{block.end_line}"] = block.proofaug_content
                                break
                        if block.state != BlockState.COMPLETED:
                            # block state is necessary to indicate whether this block succeeds
                            block.state = BlockState.SORRY_FAILED 
                            ps = init_ps # set to the state before this block
                            return ps, result
                    if len(ps2goals[ps]) == len(init_goals):
                        block.state = BlockState.COMPLETED
                        if block.level == 0:
                            assert result.get('proofStatus', None) == 'Completed', f"Observe {result.get('proofStatus', None)=} != 'Completed' in {block=}"
                            verify_cmd = to_command(block.proofaug_content, env=init_env, sorries=sorry_mode)
                            result_verify = self._send_command_to_repl(verify_cmd, timeout=step_timeout)
                            errors = compile_errors(result_verify)
                            if not is_complete(result_verify, block.proofaug_content):
                                proofaug_code = orig_header + block.proofaug_content
                                logger.warning(f"Reconstructed {proofaug_code=} is not complete with {result_verify=}, probably bug of proofaug or repl with {errors=}")
                                if require_reconstruct:
                                    block.state = BlockState.NO_RECONSTRUCT
                            else:
                                logger.debug(f"Verified the reconstructed proof {block.proofaug_content=}")
                            result['proofaug_body'] = block.proofaug_content
                            result['pa_depth'] = ProposalStructure(block.proofaug_content).depth

                    return ps, result

                # proofaug_content cannot indicate the type. it can be the original code.
                proofState, result = verify_block(block)
                errors = compile_errors(result)
                complete = block.state == BlockState.COMPLETED
                if complete:
                    success_type = 'pa_orig' if not proofaug_subst else 'proofaug'
                else:
                    success_type = 'pa_failed'
                verification_result = {
                    "state": block.state,
                    "complete": complete,
                    "errors": errors,
                    "header": orig_header,
                    "body": orig_body,
                    "depth": prop_struct.depth,
                    "pa_depth": result.get('pa_depth', None),
                    "success_type": success_type,
                    "proofaug_subst": proofaug_subst,
                    "proofaug_body": result.get('proofaug_body', None),
                    "last_result": result,
                    "hammer_count": hammer_count,
                }

        except TimeoutError as e:
            logger.debug(f"{e.__class__.__name__} {e}, timeout during verifying {code=}")
            verification_result = {
                "pass": False,
                "complete": False,
                "errors": [f"timeout during verifying {code=}"],
                "system_errors": [f"{e.__class__.__name__} {e}"],
            }
            self._clean_init_repl()

        except Exception as e:
            logger.error(f"{e.__class__.__name__} {e}, unhandled error during verifying {code=}")
            verification_result = {
                "pass": False,
                "complete": False,
                "errors": [f"exception in verifying {code=}: {e.__class__.__name__} {e}"],
                "system_errors": [traceback.format_exc()],
            }
            self._clean_init_repl()
            
        verification_result['verify_time'] = time.time() - start_time
        return verification_result
    
    def _cleanup_repl(self):
        """Clean up the REPL process and pseudo-terminal using more reliable termination"""
        # TODO: make it more reliable, fewer magic numbers
        try:
            # only cleanup if repl process is running
            if hasattr(self, 'repl_process') and self.repl_process:
                proc = self.repl_process
                if proc.poll() is None:
                    # Send SIGTERM to process group
                    if proc.pid is not None:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    
                    # Wait with timeout
                    timeout = 3  # Total wait time
                    start = time.time()
                    while proc.poll() is None and (time.time() - start) < timeout:
                        time.sleep(0.1)
                    
                    # Force kill if still running
                    if proc.poll() is None and proc.pid is not None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            logger.error(f"Process {self.idx}: ProcessLookupError when killing REPL process")
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
        finally:
            # Handle file descriptor cleanup
            if hasattr(self, 'master_fd') and self.master_fd is not None:
                try:
                    os.close(self.master_fd)
                except (OSError, TypeError) as e:
                    logger.error(f"Error closing master_fd: {str(e)}")
                self.master_fd = None
            self.repl_process = None
            self.latest_state = None
            self.restart_count = 0
            self.header_dict.clear()

    def run(self):
        """Main worker process loop - runs once per process"""
        assert self.use_pty, "non-PTY mode is no longer supported"
        init_ret = self._clean_init_repl()
        # print(f"init_ret: {init_ret}")
        if not init_ret:
            logger.error(f"Process {self.idx}: Failed to create initial REPL process, exiting")
            return

        while True:
            inputs = self.task_queue.get()
            if inputs is None:
                break
            for _, request_id, task in inputs:
                ret_code = self.repl_process.poll()
                if ret_code is not None:
                    if ret_code == 134:
                        logger.debug(f"REPL process died with code {ret_code}, restarting, most probably due to memory limit")
                    else:
                        logger.warning(f"REPL process died with code {ret_code}, probably due to wrong command")
                    if not self._clean_init_repl():
                        raise Exception(f"Process {self.idx}: Failed to restart REPL, skipping task")
                if isinstance(task, str):
                    task = dict(code=task)
                # please refer to the list of arguments in _verify_lean4_with_persistent_repl method
                
                total_timeout = task.pop('total_timeout', None)  # 从task中读取timeout，没有则为None（无限等）
                if total_timeout is not None:
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(self._verify_lean4_with_persistent_repl, **task)
                            result = future.result(timeout=total_timeout)
                    except concurrent.futures.TimeoutError:
                        result = {
                            "pass": False,
                            "complete": False,
                            "errors": [f"verification timed out after {total_timeout} seconds for code: {task.get('code', None)}"],
                            "system_errors": [f"TimeoutError: verification timed out after {total_timeout} seconds"],
                        }
                        self._clean_init_repl()  # 超时后重启REPL
                else:
                    # 没有timeout限制，直接调用
                    result = self._verify_lean4_with_persistent_repl(**task)

                with self.lock:
                    self.request_statuses[request_id] = result
                    self.last_output_time.value = time.time()
                    self.complete_count.value += 1
            self.restart_count += 1
            if self.restart_count >= self.pty_restart_count:
                self._clean_init_repl()
        self._cleanup_repl()


        

class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier', 
                 lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE,
                 use_pty=False, pty_restart_count=100, repl_path=DEFAULT_REPL_PATH):
        super().__init__(batch_size=1, name=name)
        self.use_pty = use_pty
        self.timeout = timeout
        self.pty_restart_count = pty_restart_count
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                timeout=timeout,
                memory_limit=memory_limit,
                lake_path=lake_path,
                repl_path=repl_path,
                lean_workspace=lean_workspace,
                use_pty=use_pty,
                pty_restart_count=pty_restart_count
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        logger.info(f'Launched {len(self.processes)} LeanServerProcesses in {lean_workspace}')

        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        if not self.use_pty:
            self._monitor_process.start()
    
    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            if not self.use_pty:
                kill_timeout = self.timeout + 10
            else:   # normally it does not happen
                kill_timeout = self.pty_restart_count * self.timeout + 10
            # Kill both lake and repl processes that are older than timeout
            subprocess.run(['killall', '-r', 'lake|repl', f'--older-than={int(kill_timeout)}s'], capture_output=True)
    
    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        logger.info(f'All {len(self.processes)} LeanServerProcesses stopped')
        self._running_monitor.value = False
        if not self.use_pty:
            self._monitor_process.join()
            logger.info('Monitor process stopped')
