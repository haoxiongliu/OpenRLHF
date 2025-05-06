import threading
import queue

# 从你的项目里 import
from prover.lean.verifier import Lean4ServerProcess

if __name__ == "__main__":
    # 准备一个普通的 queue、dict、lock
    task_q = queue.Queue()
    statuses = {}
    lock = threading.Lock()

    # 构造一个“子进程”对象，但不 start()
    p = Lean4ServerProcess(
        idx=0,
        task_queue=task_q,
        request_statuses=statuses,
        lock=lock,
        timeout=60,
        memory_limit=-1,
        use_pty=False
    )

    lean_code = r"""import Mathlib
theorem mathd_algebra_114 (a : ℝ) (h₀ : a = 8) :
    (16 * (a ^ 2) ^ ((1 : ℝ) / 3)) ^ ((1 : ℝ) / 3) = 4 := by
have ha : a ^ 2 = 64 := by
    rw [h₀]
    norm_num
have h1 : (a ^ 2) ^ ((1 : ℝ) / 3) = 4 := by
    rw [ha]
    have h4 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
    rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
    rw [←Real.rpow_mul]
    norm_num
    all_goals linarith
    exact h4
have h2 : 16 * (a ^ 2) ^ ((1 : ℝ) / 3) = 64 := by
    rw [h1]
    norm_num
rw [h2]
have h3 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
    rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
    rw [←Real.rpow_mul]
    norm_num
    all_goals linarith
exact h3"""

    # 
    task_q.put([
        (None, "test_req_1", {
            "code": lean_code,
            "proofaug": True,
            "sorries": "grouped"
        })
    ])
    # 用 None 标记结束
    # task_q.put(None)

    # 调用 run()（在当前进程里执行），就可以打断点、print、用 pdb 单步调试了
    p.run()
