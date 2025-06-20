#!/usr/bin/env python3
"""
Test script for the Lean Command Generator web service
"""

import requests
import json
import sys
import time

def test_service(base_url="http://localhost:8000"):
    """Test the web service with various inputs"""
    
    print(f"Testing Lean Command Generator service at {base_url}")
    print("=" * 50)
    
    # Test cases for command generation
    generate_test_cases = [
        {
            "name": "Basic command generation",
            "data": {
                "code": "theorem example : 1 + 1 = 2 := by simp"
            }
        },
        {
            "name": "With environment",
            "data": {
                "code": "rw [add_comm]",
                "env": 2
            }
        },
        {
            "name": "With proof state",
            "data": {
                "code": "simp_all",
                "proofState": 1
            }
        },
        {
            "name": "With sorries",
            "data": {
                "code": "sorry",
                "sorries": "grouped"
            }
        }
    ]
    
    # Test cases for REPL execution
    repl_test_cases = [
        {
            "name": "Simple theorem",
            "data": {
                "code": "theorem test_theorem : 1 + 1 = 2 := by norm_num"
            }
        },
        {
            "name": "Basic arithmetic",
            "data": {
                "code": "example : 2 + 3 = 5 := by norm_num"
            }
        }
    ]
    
    # Test command generation
    print("\n=== Testing Command Generation ===")
    for i, test_case in enumerate(generate_test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{base_url}/generate_command",
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print("✅ Success!")
                    print(f"Generated command: {json.dumps(result['command'], indent=2)}")
                else:
                    print(f"❌ API Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network Error: {e}")
    
    # Test REPL status
    print(f"\n=== Testing REPL Status ===")
    try:
        response = requests.get(f"{base_url}/repl_status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            if status.get("alive"):
                print("✅ REPL is alive and running!")
                print(f"PID: {status.get('pid', 'unknown')}")
            else:
                print(f"❌ REPL is not running: {status.get('message', 'unknown')}")
        else:
            print(f"❌ REPL status error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ REPL status network error: {e}")
    
    # Test REPL execution
    print(f"\n=== Testing REPL Execution ===")
    for i, test_case in enumerate(repl_test_cases, 1):
        print(f"\nREPL Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{base_url}/apply_to_repl",
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=30  # Longer timeout for REPL
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    status = "✅ Complete" if result.get("complete") else ("⚠️ Passed" if result.get("pass") else "❌ Failed")
                    print(f"{status}")
                    print(f"Verification time: {result.get('verify_time', 0):.3f}s")
                    
                    if result.get("errors"):
                        print("Errors:")
                        for error in result["errors"]:
                            print(f"  - {error.get('data', error)}")
                    
                    if result.get("infos"):
                        print("Info:")
                        for info in result["infos"]:
                            print(f"  - {info.get('data', info)}")
                else:
                    print(f"❌ REPL Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network Error: {e}")
    
    # Test API docs endpoint
    print(f"\nTesting API docs endpoint...")
    try:
        response = requests.get(f"{base_url}/api/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API docs endpoint working!")
            docs = response.json()
            print(f"Available endpoints: {list(docs.get('endpoints', {}).keys())}")
        else:
            print(f"❌ API docs error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs network error: {e}")

def wait_for_service(base_url="http://localhost:8000", max_wait=30):
    """Wait for the service to start"""
    print(f"Waiting for service to start at {base_url}...")
    
    for i in range(max_wait):
        try:
            response = requests.get(base_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print(f"❌ Service did not start within {max_wait} seconds")
    return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    if wait_for_service(base_url):
        test_service(base_url)
    else:
        print("\nTo start the service manually:")
        print("cd prover/command_generator")
        print("python start_server.py")
        sys.exit(1) 