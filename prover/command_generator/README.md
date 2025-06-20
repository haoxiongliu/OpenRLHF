# Lean Command Generator Web Service

现在发现对多进程理解有问题，等待server的东西不应该用deepseek-prover-v1.5的多进程，而是重构一个多线程/异步更好。

A FastAPI-based web service that provides a user-friendly interface for the `to_command` function from `prover/utils.py`, with integrated Lean4 REPL backend support.

## Features

- **Web Interface**: Clean, responsive web UI for generating Lean commands
- **Real-time Generation**: Instant command generation based on user input
- **REPL Backend**: Integrated Lean4ServerProcess for executing commands in a persistent REPL environment
- **Session Persistence**: Maintains REPL state across multiple commands
- **Parameter Support**: Full support for key `to_command` parameters:
  - `code`: The Lean code/command (required)
  - `env`: Environment number (optional)
  - `proofState`: Proof state number (optional)
  - `sorries`: Sorry handling (`grouped` or `individual`)
- **JSON Output**: Pretty-formatted JSON output for easy copying
- **REPL Execution**: Apply commands directly to REPL and see verification results
- **Copy Functionality**: One-click copying of generated JSON commands
- **Error Handling**: Comprehensive error handling and user feedback
- **API Endpoints**: RESTful API for programmatic access

## Installation

1. Install dependencies:
```bash
cd prover/command_generator
pip install -r requirements.txt
```

2. Or install individual packages:
```bash
pip install fastapi uvicorn[standard] jinja2 python-multipart pydantic
```

## Usage

### Start the Web Service

**Recommended method (from project root):**
```bash
# From the project root directory
python main.py start_command_generator

# With custom host and port
python main.py start_command_generator --host=127.0.0.1 --port=8080

# With auto-reload for development
python main.py start_command_generator --reload=true
```

**Alternative method (from command_generator directory):**
```bash
cd prover/command_generator

# Basic start
python start_server.py

# With custom host and port
python start_server.py --host 127.0.0.1 --port 8080

# With auto-reload for development
python start_server.py --reload
```

### Access the Service

1. **Web Interface**: Open your browser and go to `http://localhost:8000`
2. **API Documentation**: Visit `http://localhost:8000/api/docs`

### Using the Web Interface

1. Enter your Lean code in the text area
2. Optionally configure parameters (environment, proof state, etc.)
3. Choose from two actions:
   - **Generate Command**: Creates JSON command for manual use
   - **Apply to REPL**: Executes command in the Lean REPL backend
4. Copy generated JSON commands as needed
5. View REPL execution results with detailed feedback

### REPL Backend

The service maintains a persistent Lean4ServerProcess that:
- Uses PTY mode (`use_pty=True`) for reliable session management
- Maintains state across multiple commands
- Provides detailed verification results
- Handles errors gracefully with automatic recovery

### Using the API

#### Generate Command
```bash
curl -X POST "http://localhost:8000/generate_command" \
     -H "Content-Type: application/json" \
     -d '{
       "code": "example theorem",
       "env": 1
     }'
```

#### Apply to REPL
```bash
curl -X POST "http://localhost:8000/apply_to_repl" \
     -H "Content-Type: application/json" \
     -d '{
       "code": "theorem test : 1 + 1 = 2 := by simp"
     }'
```

#### Check REPL Status
```bash
curl -X GET "http://localhost:8000/repl_status"
```

## API Endpoints

- `GET /`: Main web interface
- `POST /generate_command`: Generate command from input parameters
- `POST /apply_to_repl`: Apply command to REPL backend
- `GET /repl_status`: Check REPL process status
- `GET /api/docs`: API documentation

## Examples

### Basic Command Generation
```json
{
  "code": "theorem example : 1 + 1 = 2 := by simp"
}
```

### With Environment and Proof State
```json
{
  "code": "rw [add_comm]",
  "env": 2,
  "proofState": 1
}
```

### With Sorries
```json
{
  "code": "simp_all",
  "proofState": 1,
  "sorries": "grouped"
}
```

### REPL Response Format
```json
{
  "success": true,
  "complete": true,
  "pass": true,
  "errors": [],
  "warnings": [],
  "infos": [{"data": "Goals accomplished"}],
  "sorries": [],
  "tactics": [],
  "ast": {},
  "verify_time": 0.123
}
```

## Project Structure

```
prover/command_generator/
├── app.py                 # Main FastAPI application with REPL backend
├── start_server.py        # Server startup script
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── test_service.py       # Test script
└── templates/
    └── command_generator.html  # Web interface template
```

## REPL Backend Configuration

The Lean4ServerProcess is configured with:
- **Timeout**: 60 seconds per command
- **Memory Limit**: 2GB
- **PTY Mode**: Enabled for reliable session management
- **Restart Count**: 100 (REPL restarts after 100 commands)
- **Concurrent Requests**: Limited to 2 for web service use

## Development

The service is built with:
- **FastAPI**: Modern Python web framework
- **Lean4ServerProcess**: Direct REPL backend integration
- **Uvicorn**: ASGI server
- **Jinja2**: Template engine
- **HTML/CSS/JavaScript**: Frontend interface

To modify the interface, edit `templates/command_generator.html`. The main application logic is in `app.py`.

## Integration

The service automatically imports the `to_command` function from the parent `utils.py` module and integrates with the Lean4ServerProcess from `lean/verifier.py`, ensuring it always uses the latest version of both command generation and REPL execution without requiring code duplication.

## Troubleshooting

- **REPL Not Starting**: Check that Lean4 and the REPL are properly installed
- **Timeout Errors**: Increase timeout or simplify commands
- **Memory Issues**: Reduce memory limit or restart service
- **Port Conflicts**: Use different port with `--port` argument 