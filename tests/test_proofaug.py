"""
Test suite for proofaug functionality.

This module contains test functions for testing the proof augmentation capabilities
of the Lean theorem prover integration.
"""

import sys
import os
import json
import datetime
import requests
import time
from typing import Optional, List, Dict, Any

# Add project root directory to Python path so we can import prover module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def handle_cases(test_cases: List[Dict[str, Any]], custom_assert_fn=None, host: str = "localhost", port: int = 8000, n: int = 1):
    """
    Test function for proofaug functionality with multiple test cases using lean_reward_server.
    
    Args:
        test_cases: List of dictionaries, each containing:
            - 'name': Test case name (string)
            - 'code': Lean code to test (string)
            - 'expected_success_type': Expected success type (string, default: 'proofaug')
            - 'hammer_list': List of hammers to use (list, default: ['aesop', 'my_hint', 'omega'])
            - 'pa_with_orig': Whether to include original proof (bool, default: False)
            - 'step_timeout': Timeout for the test (int, default: 180)
            - 'proofaug': Whether to enable proofaug (bool, default: True)
        custom_assert_fn: Optional custom assertion function that takes (result, test_case) and returns (bool, error_message)
        host: Hostname of lean_reward_server (default: "localhost")
        port: Port of lean_reward_server (default: 8000)
    
    Returns:
        List of test results with status for each test case
    """
    from prover.agent_utils import RewardRequest
    
    results = []
    server_url = f"http://{host}:{port}"
    
    for i, test_case in enumerate(test_cases):
        start_time = time.time()
        print(f"Running test case {i+1}/{len(test_cases)}: {test_case.get('name', f'Test_{i+1}')}")
        
        # Set up defaults
        lean_code = test_case['code']
        query = "```lean4\n" + lean_code + "\n```"
        expected_success_type = test_case.get('expected_success_type', 'proofaug')
        hammer_list = test_case.get('hammer_list', ['aesop', 'my_hint', 'omega'])
        pa_with_orig = test_case.get('pa_with_orig', False)
        step_timeout = test_case.get('step_timeout', 180)
        proofaug = test_case.get('proofaug', True)
        
        # Prepare request data for lean_reward_server
        request_data = RewardRequest(
            queries=[query]*n,  # Send single code as query
            proofaug=proofaug,
            pa_with_orig=pa_with_orig,
            hammer_list=hammer_list,
            step_timeout=step_timeout,
            require_reconstruct=True,
        ).model_dump(exclude_none=True)
        
        try:
            # Send request to lean_reward_server
            print(f"  Sending request to {server_url}/reward")
            response = requests.post(
                f"{server_url}/reward",
                json=request_data,
                timeout=step_timeout + 30  # Add extra time for HTTP timeout
            )
            response.raise_for_status()
            server_result = response.json()
            
            # Extract result for the single query (index 0)
            result = {}
            for key, values in server_result.items():
                if isinstance(values, list) and len(values) > 0:
                    result[key] = values[0]
                else:
                    result[key] = values
            
            # Map server response fields to expected format
            if 'success_types' in result:
                result['success_type'] = result['success_types']
            if 'rewards' in result:
                result['reward'] = result['rewards']
                result['success'] = result['rewards'] > 0
            if 'proofaug_codes' in result:
                result['proofaug_code'] = result['proofaug_codes']
            if 'verify_times' in result:
                result['verify_time'] = result['verify_times']
            
            # Default assertion
            test_passed = True
            error_message = ""
            
            if custom_assert_fn:
                # Use custom assertion function
                test_passed, error_message = custom_assert_fn(result, test_case)
            else:
                # Default assertion: check success_type
                if result.get('success_type') != expected_success_type:
                    test_passed = False
                    error_message = f"Expected success_type '{expected_success_type}', got '{result.get('success_type')}'"
            
            test_result = {
                'name': test_case.get('name', f'Test_{i+1}'),
                'passed': test_passed,
                'result': result,
                'error_message': error_message
            }
            elpased_time = time.time() - start_time
            if test_passed:
                print(f"✅ Test case {i+1} PASSED in {result.get('verify_time'):.5f}s and elapsed time {elpased_time:.5f}s")
            else:
                print(f"❌ Test case {i+1} FAILED: {error_message}")
                
        except requests.exceptions.RequestException as e:
            test_result = {
                'name': test_case.get('name', f'Test_{i+1}'),
                'passed': False,
                'result': None,
                'error_message': f"HTTP request failed: {str(e)}"
            }
            print(f"❌ Test case {i+1} FAILED with HTTP error: {str(e)}")
        except Exception as e:
            test_result = {
                'name': test_case.get('name', f'Test_{i+1}'),
                'passed': False,
                'result': None,
                'error_message': f"Exception occurred: {str(e)}"
            }
            print(f"❌ Test case {i+1} FAILED with exception: {str(e)}")
        
        results.append(test_result)

    # Print summary
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    print(f"\n📊 Test Summary: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed:")
        for r in results:
            if not r['passed']:
                print(f"  - {r['name']}: {r['error_message']}")
    
    # Save results to logs directory with timestamp
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proofaug_test_results_{timestamp}.json"
    filepath = os.path.join(logs_dir, filename)
    
    # Prepare data to save
    save_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'server_url': server_url,
        'total_tests': total_count,
        'passed_tests': passed_count,
        'test_results': results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Test results saved to: {filepath}")
    return results

def test_example_simple(host: str = "localhost", port: int = 8000, n: int = 1):
    """
    Simple example with just one test case.
    
    Args:
        host: Hostname of lean_reward_server (default: "localhost")
        port: Port of lean_reward_server (default: 8000)
    """
    test_cases = [
        {
            "name": "simplest example",
            "code": 'theorem foo: 2+3=5:= by simp',
            "expected_success_type": "original",
            "hammer_list": ['linarith'],
            "pa_with_orig": True,
            "step_timeout": 180,
        },
        {
            "name": "simplest with import",
            "code": 'import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem foo: 2+3=5:= by simp',
            "expected_success_type": "original",
            "hammer_list": ['linarith'],
            "pa_with_orig": True,
            "step_timeout": 180,
        },
        {
            "name": "simplest with import second time",
            "code": 'import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem foo: 2+3=5:= by simp',
            "expected_success_type": "original",
            "hammer_list": ['linarith'],
            "pa_with_orig": True,
            "step_timeout": 180,
        },
        {
            "name": "pset sorry test",
            "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem equation_solution :\n  let m : \u2124 := -1\n  let n : \u2124 := -1\n  (m^2 + n)*(m + n^2) = (m - n)^3:= by\n  sorry",
            "expected_success_type": "pa_orig",
            "hammer_list": ['simp'],
            "pa_with_orig": True,
            "step_timeout": 180,
        },
        {
            "name": "mathd_algebra_114_mixh0_v1",
            "code": 'import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem foo: 2+3=5:= by \n  have h₀ (a: ℕ): a+a = 2*a := by\n    simp\n  simp',
            "expected_success_type": "proofaug",
            "hammer_list": ['linarith', 'leanhammer_0'],
            "step_timeout": 180,
        },
        {
            "name": "mathd_algebra_114_mixh0_v1",
            "code": 'import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_114 (a : ℝ) (h₀ : a = 8) :\n    (16 * (a ^ 2) ^ ((1 : ℝ) / 3)) ^ ((1 : ℝ) / 3) = 4 := by \n  rw [h₀]\n  have h1 : (8 : ℝ) ^ 2 = (64 : ℝ) := by norm_num\n  have h2 : (64 : ℝ) = (4 : ℝ) ^ 3 := by norm_num\n  have h3 : ((8 : ℝ) ^ 2 : ℝ) ^ ((1 : ℝ) / 3) = (4 : ℝ) := by\n    rw [show (8 : ℝ) ^ 2 = (64 : ℝ) by norm_num]\n    rw [h2]\n    have h4 : ((4 : ℝ) ^ 3 : ℝ) ^ ((1 : ℝ) / 3) = (4 : ℝ) := by\n      have h5 : ((4 : ℝ) ^ 3 : ℝ) ^ ((1 : ℝ) / 3) = (4 : ℝ) ^ (3 * (1 / 3 : ℝ)) := by\n        rw [← Real.rpow_natCast, ← Real.rpow_mul]\n        ring\n        all_goals norm_num\n      rw [h5]\n      norm_num\n    rw [h4]\n  have h5 : (16 * ((4 : ℝ) ) ) ^ ((1 : ℝ) / 3) = (4 : ℝ) := by\n    have h6 : (16 * (4 : ℝ)) = (64 : ℝ) := by norm_num\n    rw [h6]\n    have h7 : ((64 : ℝ) ) ^ ((1 : ℝ) / 3) = (4 : ℝ) := by\n      have h8 : ((64 : ℝ) ) ^ ((1 : ℝ) / 3) = (4 : ℝ) := by\n        have h9 : (64 : ℝ) = (4 : ℝ) ^ (3 : ℝ) := by\n          norm_num\n        rw [h9]\n        have h10 : (( (4 : ℝ) ^ (3 : ℝ) ) ) ^ ((1 : ℝ) / 3) = (4 : ℝ) ^ (3 * (1 / 3 : ℝ)) := by\n          rw [← Real.rpow_natCast, ← Real.rpow_mul]\n          ring\n          all_goals norm_num\n        rw [h10]\n        norm_num\n      rw [h8]\n    rw [h7]\n  rw [h3]\n  rw [h5]',
            "expected_success_type": "proofaug",
            "hammer_list": ['simp_all', 'field_simp', 'linarith', 'leanhammer_0',  'norm_num', 'ring_nf', 'omega'],
            "step_timeout": 180,
        },
        {
        "name": "mathd_numbertheory_341",
        "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_numbertheory_341 (a b c : ℕ) (h₀ : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)\n    (h₁ : Nat.digits 10 (5 ^ 100 % 1000) = [c, b, a]) : a + b + c = 13 := by\n  have h2 : 5 ^ 100 % 1000 = 625 := by \n    norm_num\n  rw [h2] at h₁\n  have h3 : Nat.digits 10 625 = [5, 2, 6] := by \n    norm_num\n  rw [h3] at h₁\n  have h4 : c = 6 := by\n    simp at h₁ \n    omega\n  have h5 : b = 2 := by\n    simp at h₁ \n    omega\n  have h6 : a = 5 := by\n    simp at h₁ \n    omega\n  omega",
        "expected_success_type": "proofaug",
        "hammer_list": ['leanhammer_0'],
        "step_timeout": 180,
        },
        {
            "name": "amc12a_2020_p4",
            "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem amc12a_2020_p4 (S : Finset ℕ)\n    (h₀ : ∀ n : ℕ, n ∈ S ↔ 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ d : ℕ, d ∈ Nat.digits 10 n → Even d) ∧ 5 ∣ n) :\n    S.card = 100 := by \n  have h1 : S = Finset.filter (fun n => 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ d : ℕ, d ∈ Nat.digits 10 n → Even d) ∧ 5 ∣ n) (Finset.Icc 0 9999) := by\n    ext n \n    simp \n    <;> omega\n  rw [h1]\n  rw [Finset.filter]\n  simp \n  native_decide",
            "expected_success_type": "proofaug",
            "hammer_list": ['leanhammer_0'],
            "step_timeout": 180,
        },
        {
            "name": "mathd_algebra_188 by kimina-1.7B",
            "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_188 (σ : Equiv ℝ ℝ) (h : σ.1 2 = σ.2 2) : σ.1 (σ.1 2) = 2 := by\n  have h1 : σ.1 (σ.2 2) = 2 := by\n    apply σ.1.comp_eq_id\n  have h2 : σ.1 (σ.1 2) = σ.1 (σ.2 2) := by\n    rw [show σ.1 2 = σ.2 2 by linarith [h]]\n  rw [h2]\n  exact h1",
            "hammer_list": ['leanhammer_0'],
            "expected_success_type": "proofaug",
            "step_timeout": 180,
        },
        {
            'name': 'mathd_algebra_288 by kimina-1.7B',
            'code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_288 (x y : ℝ) (n : NNReal) (h₀ : x < 0 ∧ y < 0) (h₁ : abs y = 6)\n    (h₂ : Real.sqrt ((x - 8) ^ 2 + (y - 3) ^ 2) = 15)\n    (h₃ : Real.sqrt (x ^ 2 + y ^ 2) = Real.sqrt n) : n = 52 := by \n  have h4 : y < 0 := h₀.2\n  have h5 : abs y = -y := by\n    rw [abs_of_neg h4]\n  rw [h5] at h₁\n  have h6 : y = -6 := by linarith \n  have h7 : (x - 8) ^ 2 + (y - 3) ^ 2 = 225 := by \n    calc\n      (x - 8) ^ 2 + (y - 3) ^ 2 = (Real.sqrt ((x - 8) ^ 2 + (y - 3) ^ 2)) ^ 2 := by \n        rw [Real.sq_sqrt]\n        positivity \n      _ = 15 ^ 2 := by \n        rw [h₂]\n      _ = (225 : ℝ) := by \n        norm_num\n  rw [show y = -6 by linarith [h6]] at h7 \n  have h8 : (x - 8) ^ 2 = 144 := by nlinarith \n  have h9 : x = -4 := by \n    nlinarith [h8, h₀.1]\n  have h10 : x ^ 2 + y ^ 2 = (52 : ℝ) := by \n    rw [show x = -4 by linarith [h9], show y = -6 by linarith [h6]]\n    norm_num \n  have h11 : Real.sqrt (x ^ 2 + y ^ 2) = Real.sqrt (52 : ℝ) := by \n    rw [h10]\n    all_goals norm_num \n  have h12 : Real.sqrt n = Real.sqrt (52 : ℝ) := by \n    linarith [h₃, h11]\n  have h15 : (n : ℝ) ≥ 0 := by \n    have h16 : 0 ≤ (n : ℝ) := by \n      apply Real.sqrt_nonneg \n    linarith \n  have h16 : (n : ℝ) = (52 : ℝ) := by \n    apply Real.sqrt_inj \n    · -- Show that n ≥ 0 \n      exact_mod_cast h15 \n    · -- Show that Real.sqrt n = Real.sqrt 52 \n      exact_mod_cast h12 \n  have h17 : n = 52 := by \n    exact_mod_cast h16 \n  exact h17",
            'expected_success_type': 'proofaug',
            'hammer_list': ['leanhammer_0'],
            'step_timeout': 180,
        },
        {
            'name': 'Simple Addition',
            'code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_188 (σ : Equiv ℝ ℝ) (h : σ.1 2 = σ.2 2) : σ.1 (σ.1 2) = 2 := by \n  have h1 : σ.1 (σ.2 2) = 2 := by \n    apply σ.1\n    all_goals simp\n  have h2 : σ.1 (σ.1 2) = σ.1 (σ.2 2) := by \n    rw [h]\n  rw [h1] at h2\n  exact h2",
            'expected_success_type': 'proofaug',
            'expected_proofaug_code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_188 (σ : Equiv ℝ ℝ) (h : σ.1 2 = σ.2 2) : σ.1 (σ.1 2) = 2 := by\n  have h1 : σ.1 (σ.2 2) = 2 := by aesop\n  have h2 : σ.1 (σ.1 2) = σ.1 (σ.2 2) := by\n    rw [h]\n  rw [h1] at h2\n  exact h2",
            'pa_with_orig': False,
            'hammer_list': ['aesop', 'my_hint', 'omega'],
            'step_timeout': 180,
        },
    ]
    
    return handle_cases(test_cases, host=host, port=port, n=n)


if __name__ == "__main__":
    # You can run individual tests or all examples
    # Example usage:
    # test_example_simple()  # Uses default localhost:8000
    # test_example_simple(host="your_server_host", port=8080)  # Custom server
    import argparse
    
    parser = argparse.ArgumentParser(description="Test proofaug functionality with lean_reward_server")
    parser.add_argument("--host", default="localhost", help="Hostname of lean_reward_server (default: localhost)")
    parser.add_argument("--port", type=int, default=5000, help="Port of lean_reward_server (default: 5000)")
    parser.add_argument("-n", type=int, default=1, help="cases run in parallel")

    args = parser.parse_args()
    
    print(f"🚀 Running tests against lean_reward_server at {args.host}:{args.port}")
    test_example_simple(host=args.host, port=args.port, n=args.n) 