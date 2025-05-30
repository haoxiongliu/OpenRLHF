#!/usr/bin/env python3
"""
Test script for the ProofAug MCP Server
"""

import json
import subprocess
import sys
import asyncio
from typing import Dict, Any

class MCPClient:
    def __init__(self, server_command: list):
        self.server_command = server_command
        self.process = None
    
    async def start(self):
        """Start the MCP server process"""
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    
    async def send_request(self, method: str, params: Dict[str, Any] = None, request_id: int = 1) -> Dict[str, Any]:
        """Send a request to the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode().strip())
        return response
    
    async def close(self):
        """Close the MCP server process"""
        if self.process:
            self.process.stdin.close()
            await self.process.wait()

async def test_mcp_server():
    """Test the ProofAug MCP Server"""
    client = MCPClient(["python", "mcp_server/mcp_proofaug_server.py"])
    
    try:
        await client.start()
        print("✓ MCP Server started")
        
        # Test initialization
        response = await client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        })
        print("✓ Initialization successful")
        print(f"  Server info: {response['result']['serverInfo']}")
        
        # Test tools list
        response = await client.send_request("tools/list")
        print("✓ Tools list retrieved")
        print(f"  Available tools: {[tool['name'] for tool in response['result']['tools']]}")
        
        # Test proof analysis with sample Lean code
        sample_proof = """theorem mathd_algebra_114 (a : ℝ) (h₀ : a = 8) :
    (16 * (a ^ 2) ^ ((1 : ℝ) / 3)) ^ ((1 : ℝ) / 3) = 4 := by
  have ha : a ^ 2 = 64 := by
    rw [h₀]
    norm_num
  have h1 : (a ^ 2) ^ ((1 : ℝ) / 3) = 4 := by
    rw [ha]
    exact sorry
  rw [h1]
  norm_num"""
        
        response = await client.send_request("tools/call", {
            "name": "analyze_proof_structure",
            "arguments": {
                "proof_code": sample_proof,
                "include_details": False
            }
        })
        print("✓ Proof structure analysis successful")
        print(f"{response=}")
        # Test block extraction
        response = await client.send_request("tools/call", {
            "name": "get_proof_blocks", 
            "arguments": {
                "proof_code": sample_proof,
                "block_type": "snippet"
            }
        })
        print("✓ Proof blocks extraction successful")
        print(f"{response=}")
        
        print("\n🎉 All tests passed! MCP Server is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        await client.close()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1) 