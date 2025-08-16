# Prover

This is a module for the Lean prover. You can evaluate, or start a lean reward model server.

REMEMBER to 
```bash
lake update Hammer
lake update
lake build Hammer
lake build
```
for lean-gym to get zipperposition executable. Lean will not automatically build this target

## Installation
```bash
# git clone
git clone https://github.com/haoxiongliu/OpenRLHF.git
git submodule update --init --recursive

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python=3.12
uv pip install -r requirements_prover.txt

# (Optional) if you want to train
uv pip install -r requirements_noflash.txt 
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# install lean
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

cd lean-gym-repl
lake update && lake build
cd ../lean-gym
lake update && lake build Hammer && lake build
cd ..

# start server and test
python lean_reward_server.py # default host 5000, port 
python tests/test_proofaug.py -n 8

```

If you want to run traing, follow
```bash
# install CUDA
# https://developer.nvidia.com/cuda-toolkit-archive


```


## Prover Logger

This part explains how to use the prover logger in your code.

### Basic Usage

To use the logger in any file within the prover module:

```python
from prover.logger import logger

# Then use the logger as needed
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Log Levels

The default log level is `INFO`. This means that `DEBUG` messages will not appear unless you change the log level.

Available log levels (in increasing order of severity):
- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: An indication that something unexpected happened, or may happen in the near future
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function
- `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running

## Changing Log Level

You can change the log level at runtime:

```python
from prover.logger import set_log_level

# Change to debug level
set_log_level("DEBUG")
# Or use the logging constant
set_log_level(logging.DEBUG)
```

### Log File Location

By default, logs are written to both the console and a file at `logs/prover.log`. The log file uses rotation - when it reaches 10MB, it will be archived and a new log file will be created. Up to 5 backup log files are kept.

### Best Practices

1. **Choose the right level**: Use appropriate log levels to make filtering logs easier.
2. **Be descriptive**: Include relevant information in log messages.
3. **Structured information**: For complex data, consider using formatting or JSON:
   ```python
   logger.info(f"Processing user data: {user_id}, items: {len(items)}")
   ```
4. **Include context**: Log the context when an error or important event occurs:
   ```python
   logger.error(f"Failed to process data: {error_msg}, file: {filename}, line: {line_number}")
   ```
5. **Log exceptions with traceback**:
   ```python
   try:
      # some code that might fail
   except Exception as e:
      logger.exception(f"An error occurred: {e}")  # This includes the traceback
   ```

### Customizing the Logger

The logger is initialized when the module is imported, but you can customize it further if needed:

```python
from prover.logger import set_log_level

# Change log levels for all handlers
set_log_level("DEBUG")
```

If you need more advanced customization, you can modify the `logger.py` file. 