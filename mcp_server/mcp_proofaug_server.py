#!/usr/bin/env python3
"""
MCP Server for ProofAug Service

This server provides proof structure analysis capabilities through the Model Context Protocol (MCP).
It allows AI models to analyze Lean 4 proof structures and get detailed breakdowns of proof components.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import sys
import os

# Add the project root to the path so we can import from prover
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prover.lean.psa import ProposalStructure, Block, Snippet, BlockState

# MCP Protocol Types
@dataclass
class MCPRequest:
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str = ""
    params: Optional[Dict[str, Any]] = None

@dataclass
class MCPResponse:
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

@dataclass
class MCPError:
    code: int
    message: str
    data: Optional[Any] = None

# Error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

class ProofAugMCPServer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr)  # MCP uses stderr for logging
            ]
        )
    
    def serialize_block(self, block: Block) -> Dict[str, Any]:
        """Convert a Block object to a JSON-serializable dictionary"""
        parts = []
        for part in block.parts:
            if isinstance(part, Block):
                parts.append({
                    "type": "block",
                    "data": self.serialize_block(part)
                })
            elif isinstance(part, Snippet):
                parts.append({
                    "type": "snippet", 
                    "data": self.serialize_snippet(part)
                })
        
        return {
            "index": block.index,
            "level": block.level,
            "content": block.content,
            "proofaug_content": block.proofaug_content,
            "category": block.category,
            "state": block.state.value,
            "parts": parts,
            "statement": block.statement if hasattr(block, 'statement') and block.parts else None
        }
    
    def serialize_snippet(self, snippet: Snippet) -> Dict[str, Any]:
        """Convert a Snippet object to a JSON-serializable dictionary"""
        return {
            "content": snippet.content,
            "proofaug_content": snippet.proofaug_content,
            "category": snippet.category
        }
    
    def serialize_proposal_structure(self, ps: ProposalStructure) -> Dict[str, Any]:
        """Convert a ProposalStructure to a JSON-serializable dictionary"""
        return {
            "original_proposal": ps.proposal,
            "root": self.serialize_block(ps.root) if ps.root else None
        }
    
    async def handle_analyze_proof_structure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure of a Lean 4 proof
        
        Parameters:
        - proof_code: str - The Lean 4 proof code to analyze
        - include_details: bool - Whether to include detailed part information (default: True)
        """
        try:
            proof_code = params.get("proof_code")
            if not proof_code:
                raise ValueError("proof_code parameter is required")
            
            include_details = params.get("include_details", True)
            
            # Create ProposalStructure and analyze
            ps = ProposalStructure(proof_code)
            
            if include_details:
                result = self.serialize_proposal_structure(ps)
            else:
                # Simplified version with just basic structure info
                result = {
                    "original_proposal": ps.proposal,
                    "root_content": ps.root.content if ps.root else None,
                    "root_level": ps.root.level if ps.root else None,
                    "num_parts": len(ps.root.parts) if ps.root else 0
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing proof structure: {e}")
            raise
    
    async def handle_get_proof_blocks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get individual blocks from a proof structure
        
        Parameters:
        - proof_code: str - The Lean 4 proof code to analyze
        - block_level: int - Optional filter by block level
        - block_type: str - Optional filter by block type ("block" or "snippet")
        """
        try:
            proof_code = params.get("proof_code")
            if not proof_code:
                raise ValueError("proof_code parameter is required")
            
            block_level = params.get("block_level")
            block_type = params.get("block_type")
            
            ps = ProposalStructure(proof_code)
            blocks = []
            
            def collect_blocks(block: Block, path: str = ""):
                current_path = f"{path}.{len(blocks)}" if path else "0"
                
                # Add current block if it matches filters
                if (block_level is None or block.level == block_level) and \
                   (block_type is None or block_type == "block"):
                    blocks.append({
                        "path": current_path,
                        "type": "block",
                        "level": block.level,
                        "content": block.content,
                        "state": block.state.value
                    })
                
                # Process parts
                for i, part in enumerate(block.parts):
                    part_path = f"{current_path}.{i}"
                    if isinstance(part, Block):
                        collect_blocks(part, part_path)
                    elif isinstance(part, Snippet) and \
                         (block_type is None or block_type == "snippet"):
                        blocks.append({
                            "path": part_path,
                            "type": "snippet", 
                            "content": part.content,
                            "category": part.category
                        })
            
            if ps.root:
                collect_blocks(ps.root)
            
            return {"blocks": blocks}
            
        except Exception as e:
            self.logger.error(f"Error getting proof blocks: {e}")
            raise
    
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools"""
        return {
            "tools": [
                {
                    "name": "analyze_proof_structure",
                    "description": "Analyze the hierarchical structure of a Lean 4 proof, breaking it down into blocks and snippets",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "proof_code": {
                                "type": "string",
                                "description": "The Lean 4 proof code to analyze"
                            },
                            "include_details": {
                                "type": "boolean", 
                                "description": "Whether to include detailed part information",
                                "default": True
                            }
                        },
                        "required": ["proof_code"]
                    }
                },
                {
                    "name": "get_proof_blocks", 
                    "description": "Extract individual blocks and snippets from a proof structure with optional filtering",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "proof_code": {
                                "type": "string",
                                "description": "The Lean 4 proof code to analyze"
                            },
                            "block_level": {
                                "type": "integer",
                                "description": "Optional filter by block level"
                            },
                            "block_type": {
                                "type": "string",
                                "enum": ["block", "snippet"],
                                "description": "Optional filter by block type"
                            }
                        },
                        "required": ["proof_code"]
                    }
                }
            ]
        }
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "analyze_proof_structure":
            result = await self.handle_analyze_proof_structure(arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        elif tool_name == "get_proof_blocks":
            result = await self.handle_get_proof_blocks(arguments)
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "proofaug-mcp-server",
                "version": "1.0.0"
            }
        }
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP requests"""
        try:
            if request.method == "initialize":
                result = await self.handle_initialize(request.params or {})
            elif request.method == "tools/list":
                result = await self.handle_list_tools(request.params or {})
            elif request.method == "tools/call":
                result = await self.handle_call_tool(request.params or {})
            else:
                return MCPResponse(
                    id=request.id,
                    error=asdict(MCPError(METHOD_NOT_FOUND, f"Method not found: {request.method}"))
                )
            
            return MCPResponse(id=request.id, result=result)
            
        except Exception as e:
            self.logger.error(f"Error handling request {request.method}: {e}")
            return MCPResponse(
                id=request.id,
                error=asdict(MCPError(INTERNAL_ERROR, str(e)))
            )
    
    def serialize_response(self, response: MCPResponse) -> Dict[str, Any]:
        """Serialize MCPResponse to dict, excluding None fields"""
        result = {"jsonrpc": response.jsonrpc}
        
        if response.id is not None:
            result["id"] = response.id
        
        if response.error is not None:
            result["error"] = response.error
        elif response.result is not None:
            result["result"] = response.result
        
        return result
    
    async def run(self):
        """Main server loop"""
        self.logger.info("ProofAug MCP Server starting...")
        
        while True:
            try:
                # Read JSON-RPC message from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse request
                try:
                    request_data = json.loads(line)
                    request = MCPRequest(**request_data)
                except (json.JSONDecodeError, TypeError) as e:
                    response = MCPResponse(
                        error=asdict(MCPError(PARSE_ERROR, f"Parse error: {e}"))
                    )
                    print(json.dumps(self.serialize_response(response)), flush=True)
                    continue
                
                # Handle request
                response = await self.handle_request(request)
                
                # Send response
                print(json.dumps(self.serialize_response(response)), flush=True)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                break
        
        self.logger.info("ProofAug MCP Server shutting down...")

if __name__ == "__main__":
    server = ProofAugMCPServer()
    asyncio.run(server.run()) 