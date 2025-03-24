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

from prover.lean.ast_parser import lean4_parser
from prover.workers import ProcessScheduler
from prover.utils import AttrDict

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
            "verified_code": code,
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
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        
        self.timeout = extra_args.get('timeout', 300)
        self.memory_limit = extra_args.get('memory_limit', -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
        self.lake_path = extra_args.get('lake_path', DEFAULT_LAKE_PATH)
        self.lean_workspace = extra_args.get('lean_workspace', DEFAULT_LEAN_WORKSPACE)
        self.env_cache = {}
        self.default_header = extra_args.get('default_header', LEAN4_DEFAULT_HEADER)
        self.repl_process = None

        # A Process object will execute its run method when p.start() is called
    
    def _initialize_repl_process(self):
        """Create a REPL process using a pseudo-terminal"""
        
        self._cleanup_repl()
        try:
            self.master_fd, slave_fd = pty.openpty()
            self._set_raw_mode(self.master_fd)

            # Define memory limit setup
            def set_mem_limit():
                if self.memory_limit > 0:
                    bytes_limit = self.memory_limit * 1024 ** 3  # Convert GB to bytes
                    resource.setrlimit(
                        resource.RLIMIT_AS, 
                        (bytes_limit, bytes_limit)
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
            if self.default_header:
                self._initialize_default_header_env()
            return True
        except Exception as e:
            print(f"Process {self.idx}: Failed to initialize REPL: {str(e)}")
            traceback.print_exc()
            return False
    
    def _set_raw_mode(self, fd):
        """Set terminal to raw mode for better process interaction"""
        try:
            attrs = termios.tcgetattr(fd)
            attrs[3] = attrs[3] & ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception as e:
            print(f"Warning: Failed to set raw mode: {str(e)}")
    
    def _initialize_default_header_env(self):
        try:
            command = dict(cmd=self.default_header, allTactics=False, ast=False, tactics=False, premises=False)
            result = self._send_command_to_repl(command)
            if 'env' in result:
                self.env_cache['default'] = result['env']
            else:
                print(f"Process {self.idx}: Failed to get environment from default header")
        except Exception as e:
            print(f"Process {self.idx}: Failed to initialize default header: {str(e)}")
            traceback.print_exc()
    
    def _check_header_reuse(self, code, header=None):
        if header is None:
            header = self.default_header
        # header_lines = [line.strip() for line in self.default_header.split('\n') if line.strip()]
        # code_lines = [line.strip() for line in code.split('\n') if line.strip()]
        # return all(code_lines[i] == header_line for i, header_line in enumerate(header_lines))
        return code[:len(header)] == header

    def _strip_header_from_code(self, code, header=None):
        if header is None:
            header = self.default_header
        assert code[:len(header)] == header, "Code does not match default header"
        return code[len(header):]
    
    def _send_command_to_repl(self, command: dict):
        """Send command to REPL with enhanced message framing"""
        try:
            # Send with protocol delimiter
            os.write(self.master_fd, (json.dumps(command) + "\n\n").encode())
            
            response = b''
            start_time = time.time()
            max_size = 10 * 1024 * 1024  # 10MB safety limit
            
            while True:
                # Handle timeout
                remaining = max(0.1, self.timeout - (time.time() - start_time))
                ready, _, _ = select.select([self.master_fd], [], [], remaining)
                if not ready:
                    return {"messages": [{"data": f"REPL process timeout after {self.timeout} seconds", "severity": "error"}]}
                
                # Read chunk
                chunk = os.read(self.master_fd, 4096)
                if not chunk:  # EOF
                    break
                    
                response += chunk
                if len(response) > max_size:
                    return {"messages": [{"data": "REPL process response too large", "severity": "error"}]}
                
                # Check for protocol delimiter
                if b'\r\n\r\n' in response:
                    msg, _, _ = response.partition(b'\r\n\r\n')
                    response = msg  # Keep remaining data
                    break
                    
            try:
                response_obj = json.loads(response.decode())
            except Exception as e:
                return {"messages": [{"data": str(e), "severity": "error"}]}
            return response_obj

        except Exception as e:
            error_msg = traceback.format_exc()
            print(error_msg)
            self._cleanup_repl()
            self._initialize_repl_process()
            return {"messages": [{"data": error_msg, "severity": "error"}]}
    
    def _verify_lean4_with_persistent_repl(self, code: str, allTactics: bool=False, ast: bool=False, premises: bool=False, tactics: bool=False):
        start_time = time.time()
        system_messages = ''
        
        try:
            assert 'default' in self.env_cache, "Default header environment not cached"
            command = dict(cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
            if self._check_header_reuse(code):
                modified_code = self._strip_header_from_code(code)
                command.update(env=self.env_cache['default'], cmd=modified_code)
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
                "verified_code": code,  # Keep original code for reference
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
                            pass
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
        finally:
            # Handle file descriptor cleanup
            if hasattr(self, 'master_fd') and self.master_fd is not None:
                try:
                    os.close(self.master_fd)
                except (OSError, TypeError) as e:
                    print(f"Error closing master_fd: {str(e)}")
                self.master_fd = None
            
            self.repl_process = None
    
    def run(self):
        """Main worker process loop - runs once per process"""
        # Initial REPL creation - only happens once per worker process
        if not self._initialize_repl_process():
            print(f"Process {self.idx}: Failed to create initial REPL process, exiting")
            return
        # print(f"Process {self.idx}: Cached default header environment")

        try:
            count = 0
            while True:
                count += 1
                if count > 1:
                    self._initialize_repl_process()
                    count = 0
                inputs = self.task_queue.get()
                if inputs is None:  # Terminate signal
                    break
                
                for _, request_id, task in inputs:
                    # Only check and restart REPL if it died
                    ret_code = self.repl_process.poll()
                    if ret_code is not None:
                        print(f"Process {self.idx}: REPL process died with code {ret_code}, restarting")
                        if not self._initialize_repl_process():
                            raise Exception(f"Process {self.idx}: Failed to restart REPL, skipping task")
                    
                    # task can be a string or a dict
                    if isinstance(task, str):
                        task = dict(code=task)
                    elif not isinstance(task, dict):
                        raise Exception(f"Process {self.idx}: Invalid task type {type(task)}, skipping")
                    
                    # Use our single persistent REPL instance to verify
                    result = self._verify_lean4_with_persistent_repl(**task)
                    
                    # Handle specific errors requiring restart
                    critical_errors = [
                        'lean::exception: failed to create thread', 
                        'std::bad_alloc: std::bad_alloc',
                        'Cannot allocate memory'
                    ]
                    
                    if result.get('system_messages') and any(err in result['system_messages'] for err in critical_errors):
                        retry_start = time.time()
                        print(f"Process {self.idx}: Critical error detected, attempting REPL restart")
                        while (any(err in result['system_messages'] for err in critical_errors) and 
                              time.time() - retry_start < self.timeout):
                            self._initialize_repl_process()  # This will clean up and create new REPL
                            time.sleep(0.1)
                            result = self._verify_lean4_with_persistent_repl(**task)
                    
                    with self.lock:
                        self.request_statuses[request_id] = result
                        self.last_output_time.value = time.time()
                        self.complete_count.value += 1
        finally:
            self._cleanup_repl()


class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier', 
                 lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE,
                 default_header=LEAN4_DEFAULT_HEADER):
        super().__init__(batch_size=1, name=name)
        
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                    lake_path=lake_path,
                    lean_workspace=lean_workspace,
                    default_header=default_header,
                )
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f'Launched {len(self.processes)} LeanServerProcesses')

    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        print(f'All {len(self.processes)} LeanServerProcesses stopped')


if __name__ == '__main__':
    code = open('mathlib4/.lake/packages/REPL/test/aime_1983_p9.in').read()
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([dict(code=code, ast=True, tactics=True)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    print(outputs_list)
