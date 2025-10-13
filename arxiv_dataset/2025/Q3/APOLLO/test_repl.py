from prover.lean.verifier import verify_lean4_file

code = '''import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem test
  (x y z w : ℕ)
  (ht : 1 < x ∧ 1 < y ∧ 1 < z)
  (hw : 0 < w)
  (h0 : Real.logb x w = 24)
  (h1 : Real.logb y w = 40)
  (h2 : Real.logb (x * y * z) w = 12) :
  Real.logb z w = 60 := by
  sorry
'''

result = verify_lean4_file(code, timeout=30)

print(result)