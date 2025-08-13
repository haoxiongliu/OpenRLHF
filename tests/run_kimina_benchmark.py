"""
like test_proofaug, but less output info, just focus on the pass rate and the time cost
using Goedel-LM/Lean-workbook-proofs, whose data has full_proof and problem_id two fields
default do not use proofaug.
"""

import sys
import os
import json
import time
import datetime
import requests
from typing import Optional, List, Dict, Any
from datasets import load_dataset

# Add project root directory to Python path so we can import prover module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prover.agent_utils import RewardRequest


def run_kimina_benchmark(
    dataset_name: str = "Goedel-LM/Lean-workbook-proofs",
    host: str = "localhost", 
    port: int = 8000,
    num_samples: Optional[int] = None,
    proofaug: bool = False,
    step_timeout: float = 180.0,
    hammer_list: List[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run benchmark on Lean-workbook-proofs dataset focusing on pass rate and time cost.
    
    Args:
        dataset_name: HuggingFace dataset name (default: "Goedel-LM/Lean-workbook-proofs")
        host: Hostname of lean_reward_server (default: "localhost")
        port: Port of lean_reward_server (default: 8000)
        num_samples: Number of samples to test (default: None for all)
        proofaug: Whether to enable proofaug (default: False)
        step_timeout: Timeout for each test (default: 180.0)
        hammer_list: List of hammers to use (default: ['simp'])
        output_file: Path to save results (default: auto-generated)
    
    Returns:
        Dictionary containing benchmark results
    """
    if hammer_list is None:
        hammer_list = ['simp']
    
    print(f"🚀 Loading dataset: {dataset_name}")
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split="train")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"📊 Dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {str(e)}")
        print("💡 Trying to load from local cache or using example data...")
        # Fallback to example data if dataset loading fails
        dataset = [
            {
                "problem_id": "example_1", 
                "full_proof": "import Mathlib\n\ntheorem example: 2 + 3 = 5 := by simp"
            },
            {
                "problem_id": "example_2",
                "full_proof": "import Mathlib\n\ntheorem foo: ∀ n : ℕ, n + 0 = n := by simp"
            }
        ]
        if num_samples:
            dataset = dataset[:num_samples]
    
    server_url = f"http://{host}:{port}"
    print(f"🔗 Testing against lean_reward_server at {server_url}")
    print(f"⚙️  Proofaug: {'Enabled' if proofaug else 'Disabled'}")
    print(f"🔨 Hammers: {hammer_list}")
    print(f"⏱️  Timeout: {step_timeout}s")
    print("=" * 60)
    
    # Track results
    results = []
    total_time = 0.0
    passed_count = 0
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        problem_id = item.get("problem_id", f"problem_{i}")
        full_proof = item.get("full_proof", "")
        
        if not full_proof.strip():
            print(f"⚠️  Skipping {problem_id}: empty proof")
            continue
            
        # print(f"[{i+1}/{len(dataset)}] Testing {problem_id}...", end=" ")
        
        # Prepare Lean code query
        query = f"```lean4\n{full_proof}\n```"
        
        # Prepare request
        request_data = RewardRequest(
            queries=[query],
            proofaug=proofaug,
            hammer_list=hammer_list,
            step_timeout=step_timeout,
            require_reconstruct=True,
        ).model_dump(exclude_none=True)
        
        # Time the request
        request_start = time.time()
        
        try:
            response = requests.post(
                f"{server_url}/reward",
                json=request_data,
                timeout=step_timeout + 30
            )
            response.raise_for_status()
            server_result = response.json()
            
            request_time = time.time() - request_start
            total_time += request_time
            
            # Extract results
            success = False
            verify_time = None
            success_type = None
            
            if 'rewards' in server_result and len(server_result['rewards']) > 0:
                reward = server_result['rewards'][0]
                success = reward > 0
                
            if 'verify_times' in server_result and len(server_result['verify_times']) > 0:
                verify_time = server_result['verify_times'][0]
                
            if 'success_types' in server_result and len(server_result['success_types']) > 0:
                success_type = server_result['success_types'][0]
            
            if success:
                passed_count += 1
                print(f"✅ PASS ({request_time:.2f}s)")
            else:
                print(f"❌ FAIL ({request_time:.2f}s)")
                
            # Store minimal result info
            results.append({
                "problem_id": problem_id,
                "success": success,
                "request_time": request_time,
                "verify_time": verify_time,
                "success_type": success_type
            })
            
        except Exception as e:
            request_time = time.time() - request_start
            total_time += request_time
            print(f"❌ ERROR ({request_time:.2f}s): {str(e)[:50]}...")
            
            results.append({
                "problem_id": problem_id,
                "success": False,
                "request_time": request_time,
                "verify_time": None,
                "success_type": None,
                "error": str(e)
            })
    
    # Calculate final metrics
    total_runtime = time.time() - start_time
    pass_rate = (passed_count / len(results)) * 100 if results else 0
    avg_time_per_case = total_time / len(results) if results else 0
    
    print("=" * 60)
    print(f"📈 BENCHMARK RESULTS:")
    print(f"   Total samples: {len(results)}")
    print(f"   Passed: {passed_count}")
    print(f"   Pass rate: {pass_rate:.1f}%")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg time per case: {avg_time_per_case:.2f}s")
    print(f"   Total runtime: {total_runtime:.2f}s")
    
    # Prepare summary
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset_name,
        "server_url": server_url,
        "config": {
            "proofaug": proofaug,
            "hammer_list": hammer_list,
            "step_timeout": step_timeout,
            "num_samples": len(results)
        },
        "metrics": {
            "total_samples": len(results),
            "passed_count": passed_count,
            "pass_rate_percent": pass_rate,
            "total_time_seconds": total_time,
            "avg_time_per_case_seconds": avg_time_per_case,
            "total_runtime_seconds": total_runtime
        },
        "detailed_results": results
    }
    
    # Save results
    if output_file is None:
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        proofaug_suffix = "_proofaug" if proofaug else "_no_proofaug"
        output_file = os.path.join(logs_dir, f"kimina_benchmark_{timestamp}{proofaug_suffix}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    
    return summary


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Kimina benchmark on Lean-workbook-proofs dataset"
    )
    parser.add_argument(
        "--dataset", 
        default="Goedel-LM/Lean-workbook-proofs",
        help="HuggingFace dataset name (default: Goedel-LM/Lean-workbook-proofs)"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Hostname of lean_reward_server (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port of lean_reward_server (default: 8000)"
    )
    parser.add_argument(
        "-n", "--num-samples", 
        type=int, 
        default=None,
        help="Number of samples to test (default: all)"
    )
    parser.add_argument(
        "--proofaug", 
        action="store_true", 
        default=False,
        help="Enable proofaug (default: False as specified in docstring)"
    )
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=180.0,
        help="Timeout for each test in seconds (default: 180.0)"
    )
    parser.add_argument(
        "--hammers", 
        nargs="+", 
        default=["simp"],
        help="List of hammers to use (default: ['simp'])"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file path (default: auto-generated in logs/)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Quick test with only 5 samples"
    )
    
    args = parser.parse_args()
    
    # Handle quick test mode
    if args.quick:
        args.num_samples = 5
        print("🚀 Running in quick test mode (5 samples)")
    
    print(f"🔧 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Server: {args.host}:{args.port}")
    print(f"   Samples: {args.num_samples or 'all'}")
    print(f"   Proofaug: {args.proofaug}")
    print(f"   Timeout: {args.timeout}s")
    print(f"   Hammers: {args.hammers}")
    print()
    
    try:
        summary = run_kimina_benchmark(
            dataset_name=args.dataset,
            host=args.host,
            port=args.port,
            num_samples=args.num_samples,
            proofaug=args.proofaug,
            step_timeout=args.timeout,
            hammer_list=args.hammers,
            output_file=args.output
        )
        
        print("\n🎯 Summary:")
        print(f"   Pass rate: {summary['metrics']['pass_rate_percent']:.1f}%")
        print(f"   Avg time: {summary['metrics']['avg_time_per_case_seconds']:.2f}s")
        print(f"   Total time: {summary['metrics']['total_time_seconds']:.2f}s")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
