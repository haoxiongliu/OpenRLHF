from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import sys
import os
import multiprocessing as mp
import time
from typing import Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import from prover module (when running from root directory)
from prover.utils import to_command, DEFAULT_LEAN_WORKSPACE, DEFAULT_LAKE_PATH, DEFAULT_REPL_PATH
from prover.lean.verifier import Lean4ServerProcess


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global repl_server  # 改个名字，更清楚表示这不是真正的"进程"
    
    # 创建对象（但不当作进程使用）
    repl_server = Lean4ServerProcess(
        idx=0,
        task_queue=None,  # 不需要
        request_statuses=None,  # 不需要
        lock=None,  # 不需要
        timeout=60,
        memory_limit=10,  # 10GB limit
        lake_path=DEFAULT_LAKE_PATH,
        repl_path=DEFAULT_REPL_PATH,
        lean_workspace=DEFAULT_LEAN_WORKSPACE,
        use_pty=True,
        pty_restart_count=100
    )
    
    # 直接初始化REPL（在当前进程中）
    repl_server._clean_init_repl()
    
    # 现在可以直接访问属性了！
    print(f"REPL master_fd: {repl_server.master_fd}")  # ✅ 可以访问
    print(f"REPL subprocess PID: {repl_server.repl_process.pid}")  # ✅ 可以访问
    
    # Store references in app state
    app.state.repl_server = repl_server
    
    print("Lean4Server initialized successfully")
    yield
    
    # Shutdown - 直接调用清理方法
    if repl_server:
        repl_server._cleanup_repl()
        print("Lean4Server terminated")

app = FastAPI(
    title="Lean Command Generator", 
    description="Web service for generating Lean commands with REPL backend",
    lifespan=lifespan
)

# Set up templates directory
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

class CommandRequest(BaseModel):
    code: str
    env: Optional[int] = None
    proofState: Optional[int] = None
    sorries: Optional[str] = None

class REPLRequest(BaseModel):
    code: str
    env: Optional[int] = None
    proofState: Optional[int] = None
    sorries: Optional[str] = None



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("command_generator.html", {"request": request})

@app.post("/generate_command")
async def generate_command(request: CommandRequest):
    """Generate command based on input parameters"""
    try:
        # Call the to_command function from utils
        command = to_command(
            code=request.code,
            env=request.env,
            proofState=request.proofState,
            sorries=request.sorries,
            verbose=True  # Always verbose for web display
        )
        
        # Format the command as JSON string
        command_json = json.dumps(command, ensure_ascii=False)
        
        return JSONResponse({
            "success": True,
            "command": command,
            "command_json": command_json
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)

@app.post("/apply_to_repl")
async def apply_to_repl(request: REPLRequest, app_request: Request):
    """Apply command to REPL environment"""
    try:
        repl_server = app_request.app.state.repl_server
        
        # 直接检查REPL子进程状态
        if not repl_server.repl_process or repl_server.repl_process.poll() is not None:
            return JSONResponse({
                "success": False,
                "error": "REPL subprocess is not running"
            }, status_code=500)
        
        # 构造命令
        command = repl_server._to_command(
            code=request.code,
            env=request.env,
            proofState=request.proofState,
            sorries=request.sorries,
            verbose=True
        )
        
        result = repl_server._send_command_to_repl(command)
        return JSONResponse({
            "command": command,
            "result": result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"REPL execution failed: {str(e)}"
        }, status_code=500)

@app.get("/repl_status")
async def get_repl_status(app_request: Request):
    """Get current REPL status"""
    try:
        repl_server = app_request.app.state.repl_server
        
        # 检查REPL子进程状态
        if not repl_server.repl_process:
            alive = False
            pid = None
        else:
            alive = repl_server.repl_process.poll() is None
            pid = repl_server.repl_process.pid
        
        return JSONResponse({
            "alive": alive,
            "message": "REPL subprocess is running" if alive else "REPL subprocess is not running",
            "pid": pid,
            "master_fd": repl_server.master_fd
        })
        
    except Exception as e:
        return JSONResponse({
            "alive": False,
            "error": str(e)
        })

@app.get("/api/docs")
async def get_api_docs():
    """Get API documentation"""
    return {
        "endpoints": {
            "/": "Main web interface",
            "/generate_command": "POST endpoint to generate commands",
            "/apply_to_repl": "POST endpoint to apply commands to REPL",
            "/repl_status": "GET endpoint to check REPL status",
            "/api/docs": "This documentation"
        },
        "command_parameters": {
            "code": "The Lean code/command (required)",
            "env": "Environment number (optional, integer)",
            "proofState": "Proof state number (optional, integer)",
            "sorries": "Sorry handling ('grouped' or 'individual', optional)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 