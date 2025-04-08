# Lean4 verifier implementation based on DeepSeek-Prover-V1.5
import os
import time
import json
import ctypes
import traceback
import subprocess
import multiprocessing as mp
import threading
import fcntl
import select
import pty
import termios
import signal
import resource
import errno
import re

from prover.lean.ast_parser import lean4_parser
from prover.workers import ProcessScheduler
from prover.logger import logger
from prover.utils import remove_lean_comments

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'
LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def verify_lean4_file(code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, 
                      verbose=False, timeout=300, allTactics=False, ast=False, premises=False, tactics=False):
    """Standalone verification function that creates a new repl process for each verification."""
    command = dict(cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ''
    try:
        outputs = subprocess.run(
            [lake_path, "exe", 'repl'], 
            input=message_str + "\r\n\r\n", 
            capture_output=True, 
            text=True, 
            cwd=lean_workspace, 
            timeout=timeout
        )
        result = json.loads(outputs.stdout)
        ast_results = lean4_parser(code, result['ast']) if 'ast' in result and result['ast'] else {}
        result = {
            "sorries": result.get('sorries', []), 
            "tactics": result.get('tactics', []),
            "errors": [m for m in result.get('messages', []) if m['severity'] == 'error'],
            "warnings": [m for m in result.get('messages', []) if m['severity'] == 'warning'],
            "infos": [m for m in result.get('messages', []) if m['severity'] == 'info'],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            # "verified_code": code,
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any(
            "declaration uses 'sorry'" in w['data'] or 'failed' in w['data'] for w in result['warnings']
        )
    except Exception:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages
        }
    result['verify_time'] = time.time() - start_time
    return result

class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, timeout=300, memory_limit=-1, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, default_header=None, use_pty=False, pty_restart_count=3):
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
        self.lean_workspace = lean_workspace
        self.header_dict = {}  # Dictionary to store different headers and their environments
        self.use_pty = use_pty
        self.pty_restart_count = pty_restart_count
        self.repl_process = None
        
    def _initialize_repl_process(self):
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

            self.repl_process = subprocess.Popen(
                [self.lake_path, "exe", 'repl'],
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
            logger.error(f"Process {self.idx}: Failed to initialize REPL: {str(e)}", stack_info=True)
            return False
    
    def _set_raw_mode(self, fd):
        """Set terminal to raw mode for better process interaction"""
        try:
            attrs = termios.tcgetattr(fd)
            attrs[3] = attrs[3] & ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception as e:
            logger.warning(f"Warning: Failed to set raw mode: {str(e)}")
    
    def _split_header_body(self, code):
        """Split the code into header and body. None if no header found."""
        # TODO: add support for more keywords, or other heuristics
        clean_code = remove_lean_comments(code)
        match = re.search(r'\b(theorem|example|def|abbrev)\b', clean_code)
        if match:
            header, body = clean_code[:match.start()].strip(), clean_code[match.start():].strip()
        else:
            header, body = None, clean_code
        return header, body
    
    def _initialize_header_env(self, header):
        """Initialize the environment for a given header"""
        command = dict(cmd=header, allTactics=False, ast=False, tactics=False, premises=False)
        result = self._send_command_to_repl(command)
        if 'env' not in result:
            messages = result.get('messages', [])   
            logger.error(f"Process {self.idx}: Failed to initialize {header=} with {messages=}", stack_info=False)
            return None
        else:
            self.header_dict[header] = result['env']
            return result['env']

    
    def _send_command_to_repl(self, command: dict):
        """Send command to REPL with enhanced message framing"""
        try:
            # Send with protocol delimiter
            os.write(self.master_fd, (json.dumps(command) + "\n\n").encode())
            
            response = b''
            start_time = time.time()
            max_size = 10 * 1024 * 1024  # 10MB safety limit

            while True:
                ready, _, _ = select.select([self.master_fd], [], [], 0.0)
                if not ready:
                    break
                last_chunk = os.read(self.master_fd, 4096)  # Discard any buffered data
                logger.warning(f"Non-read buffer data: {last_chunk}")
                
            while True:
                # Handle timeout
                remaining = max(0.1, self.timeout - (time.time() - start_time))
                ready, _, _ = select.select([self.master_fd], [], [], remaining)
                if not ready:
                    # seems to be the cause of misplaced timeout
                    raise TimeoutError(f"REPL process timeout after {self.timeout} seconds")
                
                # Read chunk
                try:
                    chunk = os.read(self.master_fd, 4096)
                except (OSError, IOError) as e:
                    # Handle common errors that can occur even after select
                    if e.errno in (errno.EBADF, errno.EINVAL):  # Bad file descriptor or Invalid argument
                        return {"messages": [{"data": f"REPL process file descriptor error: {str(e)}", "severity": "error"}]}
                    elif e.errno == errno.EINTR:  # Interrupted by signal
                        continue  # Retry the read
                    else:
                        return {"messages": [{"data": f"Unexpected error reading from REPL: {str(e)}", "severity": "error"}]}
                if not chunk:  # EOF
                    break
                    
                response += chunk
                if len(response) > max_size:
                    return {"messages": [{"data": "REPL process response too large", "severity": "error"}]}

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
                return response_obj
            except Exception as e:
                return {"messages": [{"data": str(e), "severity": "error"}]}
            
        except Exception as e:
            self._cleanup_repl()
            self._initialize_repl_process()
            return {"messages": [{"data": str(e), "severity": "error"}]}
    
    def _verify_lean4_with_persistent_repl(self, code: str, allTactics: bool=False, ast: bool=False, premises: bool=False, tactics: bool=False, proof_aug: bool=False):
        start_time = time.time()
        system_messages = ''
        
        if proof_aug:
            # proof_aug is only supported in Pty mode
            raise NotImplementedError("ProofAug is not supported yet")

        try:
            header, body = self._split_header_body(code)
            command = dict(cmd=body, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
            
            # Check if we already have an environment for this header
            if header in self.header_dict:
                # Use the cached environment
                command.update(env=self.header_dict[header])
            elif header:
                # If this is a new header, initialize its environment and cache it
                # Exception handled in _initialize_header_env
                env = self._initialize_header_env(header)
                if env:
                    command.update(env=env)
            
            result = self._send_command_to_repl(command)
            
            ast_results = lean4_parser(code, result['ast']) if 'ast' in result and result['ast'] else {}
            verification_result = {
                "sorries": result.get('sorries', []), 
                "tactics": result.get('tactics', []),
                "errors": [m for m in result.get('messages', []) if m['severity'] == 'error'],
                "warnings": [m for m in result.get('messages', []) if m['severity'] == 'warning'],
                "infos": [m for m in result.get('messages', []) if m['severity'] == 'info'],
                "system_messages": system_messages,
                "system_errors": None,
                "ast": ast_results,
                # "verified_code": code,  # Keep original code for reference
            }
            verification_result['pass'] = not verification_result['errors']
            verification_result['complete'] = (
                verification_result['pass'] and 
                not verification_result['sorries'] and 
                not any("declaration uses 'sorry'" in w['data'] or 'failed' in w['data'] 
                      for w in verification_result['warnings'])
            )
            
        except Exception:
            verification_result = {
                "pass": False,
                "complete": False,
                "system_errors": traceback.format_exc(),
                "system_messages": system_messages
            }
            self._cleanup_repl()
            self._initialize_repl_process()
            
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
                    if proc.pid:
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
                    if proc.poll() is None and proc.pid:
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
            self.header_dict.clear()

    def run(self):
        """Main worker process loop - runs once per process"""
        if self.use_pty:
            if not self._initialize_repl_process():
                logger.error(f"Process {self.idx}: Failed to create initial REPL process, exiting")
                return

            count = 0
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
                            logger.error(f"REPL process died with code {ret_code}, unknown cause")
                        if not self._initialize_repl_process():
                            raise Exception(f"Process {self.idx}: Failed to restart REPL, skipping task")
                    if isinstance(task, str):
                        task = dict(code=task)
                    elif not isinstance(task, dict):
                        raise Exception(f"Process {self.idx}: Invalid task type {type(task)}, skipping")
                    result = self._verify_lean4_with_persistent_repl(**task)

                    with self.lock:
                        self.request_statuses[request_id] = result
                        self.last_output_time.value = time.time()
                        self.complete_count.value += 1
                count += 1
                if count >= self.pty_restart_count:
                    self._initialize_repl_process()
                    count = 0
        else:
            # Non-PTY mode: use verify_lean4_file directly and bypass persistent REPL and header checks
            while True:
                inputs = self.task_queue.get()
                if inputs is None:
                    break
                for _, request_id, task in inputs:
                    if isinstance(task, str):
                        task = dict(code=task)
                    if 'timeout' not in task:
                        task['timeout'] = self.timeout
                    # Directly call verify_lean4_file without any REPL state or header mechanism
                    result = verify_lean4_file(code=task['code'], lake_path=self.lake_path, lean_workspace=self.lean_workspace, timeout=self.timeout, allTactics=task.get('allTactics', False), ast=task.get('ast', False), premises=task.get('premises', False), tactics=task.get('tactics', False))
                    with self.lock:
                        self.request_statuses[request_id] = result
                        self.last_output_time.value = time.time()
                        self.complete_count.value += 1

        self._cleanup_repl()

class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier', 
                 lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE,
                 default_header=None, use_pty=False, pty_restart_count=3):
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
                lean_workspace=lean_workspace,
                default_header=default_header,
                use_pty=use_pty,
                pty_restart_count=pty_restart_count
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        logger.info(f'Launched {len(self.processes)} LeanServerProcesses')

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


if __name__ == '__main__':
    code = open('mathlib4/.lake/packages/REPL/test/aime_1983_p9.in').read()
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([dict(code=code, ast=True, tactics=True)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    print(outputs_list)
