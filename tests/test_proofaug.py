"""
Test suite for proofaug functionality.

This module contains test functions for testing the proof augmentation capabilities
of the Lean theorem prover integration.
"""

import sys
import os
import threading
import queue
import json
import datetime

# Add project root directory to Python path so we can import prover module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def handle_cases(test_cases, custom_assert_fn=None):
    """
    Test function for proofaug functionality with multiple test cases.
    
    Args:
        test_cases: List of dictionaries, each containing:
            - 'name': Test case name (string)
            - 'code': Lean code to test (string)
            - 'expected_success_type': Expected success type (string, default: 'proofaug')
            - 'hammer_list': List of hammers to use (list, default: ['bound', 'aesop', 'nlinarith', 'simp_all', 'field_simp', 'omega', 'my_hint'])
            - 'pa_with_orig': Whether to include original proof (bool, default: False)
            - 'step_timeout': Timeout for the test (int, default: 180)
            - 'memory_limit': Memory limit (int, default: 20)
        custom_assert_fn: Optional custom assertion function that takes (result, test_case) and returns (bool, error_message)
    
    Returns:
        List of test results with status for each test case
    """
    from prover.lean.verifier import Lean4ServerProcess
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"Running test case {i+1}/{len(test_cases)}: {test_case.get('name', f'Test_{i+1}')}")
        
        # Set up defaults
        lean_code = test_case['code']
        expected_success_type = test_case.get('expected_success_type', 'proofaug')
        hammer_list = test_case.get('hammer_list', ['aesop', 'my_hint', 'omega'])
        pa_with_orig = test_case.get('pa_with_orig', False)
        step_timeout = test_case.get('step_timeout', 180)
        memory_limit = test_case.get('memory_limit', 20)
        proofaug = test_case.get('proofaug', True)
        
        # Prepare queue, dict, lock
        task_q = queue.Queue()
        statuses = {}
        lock = threading.Lock()

        # Create Lean4ServerProcess
        p = Lean4ServerProcess(
            idx=0,
            task_queue=task_q,
            request_statuses=statuses,
            lock=lock,
            memory_limit=memory_limit,
            use_pty=True
        )
        
        # Put task in queue
        task_q.put([
            (None, f"test_req_{i+1}", {
                "code": lean_code,
                "hammer_list": hammer_list,
                "pa_with_orig": pa_with_orig,
                "step_timeout": step_timeout,
                "proofaug": proofaug,
            })
        ])
        
        try:
            # Run the process
            p.run()
            result = task_q.get()
            
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
            
            if test_passed:
                print(f"✅ Test case {i+1} PASSED")
            else:
                print(f"❌ Test case {i+1} FAILED: {error_message}")
                
        except Exception as e:
            test_result = {
                'name': test_case.get('name', f'Test_{i+1}'),
                'passed': False,
                'result': None,
                'error_message': f"Exception occurred: {str(e)}"
            }
            print(f"❌ Test case {i+1} FAILED with exception: {str(e)}")
        
        results.append(test_result)
    task_q.put(None)

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
        'total_tests': total_count,
        'passed_tests': passed_count,
        'test_results': results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    return results

def test_example_simple():
    """
    Simple example with just one test case.
    """
    test_cases = [
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
        "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_numbertheory_341 (a b c : \u2115) (h\u2080 : a \u2264 9 \u2227 b \u2264 9 \u2227 c \u2264 9)\n    (h\u2081 : Nat.digits 10 (5 ^ 100 % 1000) = [c, b, a]) : a + b + c = 13 := by\n  have h2 : 5 ^ 100 % 1000 = 625 := by \n    norm_num\n  rw [h2] at h\u2081\n  have h3 : Nat.digits 10 625 = [5, 2, 6] := by \n    norm_num\n  rw [h3] at h\u2081\n  have h4 : c = 6 := by\n    simp at h\u2081 \n    omega\n  have h5 : b = 2 := by\n    simp at h\u2081 \n    omega\n  have h6 : a = 5 := by\n    simp at h\u2081 \n    omega\n  omega",
        "expected_success_type": "proofaug",
        "hammer_list": ['leanhammer_0'],
        "step_timeout": 180,
        },
        {
            "name": "amc12a_2020_p4",
            "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem amc12a_2020_p4 (S : Finset \u2115)\n    (h\u2080 : \u2200 n : \u2115, n \u2208 S \u2194 1000 \u2264 n \u2227 n \u2264 9999 \u2227 (\u2200 d : \u2115, d \u2208 Nat.digits 10 n \u2192 Even d) \u2227 5 \u2223 n) :\n    S.card = 100 := by \n  have h1 : S = Finset.filter (fun n => 1000 \u2264 n \u2227 n \u2264 9999 \u2227 (\u2200 d : \u2115, d \u2208 Nat.digits 10 n \u2192 Even d) \u2227 5 \u2223 n) (Finset.Icc 0 9999) := by\n    ext n \n    simp \n    <;> omega\n  rw [h1]\n  rw [Finset.filter]\n  simp \n  native_decide",
            "expected_success_type": "proofaug",
            "hammer_list": ['leanhammer_0'],
            "step_timeout": 180,
        },
        {
            "name": "mathd_algebra_188 by kimina-1.7B",
            "code": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_188 (\u03c3 : Equiv \u211d \u211d) (h : \u03c3.1 2 = \u03c3.2 2) : \u03c3.1 (\u03c3.1 2) = 2 := by\n  have h1 : \u03c3.1 (\u03c3.2 2) = 2 := by\n    apply \u03c3.1.comp_eq_id\n  have h2 : \u03c3.1 (\u03c3.1 2) = \u03c3.1 (\u03c3.2 2) := by\n    rw [show \u03c3.1 2 = \u03c3.2 2 by linarith [h]]\n  rw [h2]\n  exact h1",
            "hammer_list": ['leanhammer_0'],
            "expected_success_type": "proofaug",
            "step_timeout": 180,
        },
        {
            'name': 'mathd_algebra_288 by kimina-1.7B',
            'code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_288 (x y : \u211d) (n : NNReal) (h\u2080 : x < 0 \u2227 y < 0) (h\u2081 : abs y = 6)\n    (h\u2082 : Real.sqrt ((x - 8) ^ 2 + (y - 3) ^ 2) = 15)\n    (h\u2083 : Real.sqrt (x ^ 2 + y ^ 2) = Real.sqrt n) : n = 52 := by \n  have h4 : y < 0 := h\u2080.2\n  have h5 : abs y = -y := by\n    rw [abs_of_neg h4]\n  rw [h5] at h\u2081\n  have h6 : y = -6 := by linarith \n  have h7 : (x - 8) ^ 2 + (y - 3) ^ 2 = 225 := by \n    calc\n      (x - 8) ^ 2 + (y - 3) ^ 2 = (Real.sqrt ((x - 8) ^ 2 + (y - 3) ^ 2)) ^ 2 := by \n        rw [Real.sq_sqrt]\n        positivity \n      _ = 15 ^ 2 := by \n        rw [h\u2082]\n      _ = (225 : \u211d) := by \n        norm_num\n  rw [show y = -6 by linarith [h6]] at h7 \n  have h8 : (x - 8) ^ 2 = 144 := by nlinarith \n  have h9 : x = -4 := by \n    nlinarith [h8, h\u2080.1]\n  have h10 : x ^ 2 + y ^ 2 = (52 : \u211d) := by \n    rw [show x = -4 by linarith [h9], show y = -6 by linarith [h6]]\n    norm_num \n  have h11 : Real.sqrt (x ^ 2 + y ^ 2) = Real.sqrt (52 : \u211d) := by \n    rw [h10]\n    all_goals norm_num \n  have h12 : Real.sqrt n = Real.sqrt (52 : \u211d) := by \n    linarith [h\u2083, h11]\n  have h15 : (n : \u211d) \u2265 0 := by \n    have h16 : 0 \u2264 (n : \u211d) := by \n      apply Real.sqrt_nonneg \n    linarith \n  have h16 : (n : \u211d) = (52 : \u211d) := by \n    apply Real.sqrt_inj \n    \u00b7 -- Show that n \u2265 0 \n      exact_mod_cast h15 \n    \u00b7 -- Show that Real.sqrt n = Real.sqrt 52 \n      exact_mod_cast h12 \n  have h17 : n = 52 := by \n    exact_mod_cast h16 \n  exact h17",
            'expected_success_type': 'proofaug',
            'hammer_list': ['leanhammer_0'],
            'step_timeout': 180,
        },
        {
            'name': 'Simple Addition',
            'code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_188 (\u03c3 : Equiv \u211d \u211d) (h : \u03c3.1 2 = \u03c3.2 2) : \u03c3.1 (\u03c3.1 2) = 2 := by \n  have h1 : \u03c3.1 (\u03c3.2 2) = 2 := by \n    apply \u03c3.1\n    all_goals simp\n  have h2 : \u03c3.1 (\u03c3.1 2) = \u03c3.1 (\u03c3.2 2) := by \n    rw [h]\n  rw [h1] at h2\n  exact h2",
            'expected_success_type': 'proofaug',
            'expected_proofaug_code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\ntheorem mathd_algebra_188 (\u03c3 : Equiv \u211d \u211d) (h : \u03c3.1 2 = \u03c3.2 2) : \u03c3.1 (\u03c3.1 2) = 2 := by\n  have h1 : \u03c3.1 (\u03c3.2 2) = 2 := by aesop\n  have h2 : \u03c3.1 (\u03c3.1 2) = \u03c3.1 (\u03c3.2 2) := by\n    rw [h]\n  rw [h1] at h2\n  exact h2",
            'pa_with_orig': False,
            'hammer_list': ['aesop', 'my_hint', 'omega'],
            'step_timeout': 180,
        },
    ]
    
    return handle_cases(test_cases)


if __name__ == "__main__":
    # You can run individual tests or all examples
    # run_all_examples()
    test_example_simple() 