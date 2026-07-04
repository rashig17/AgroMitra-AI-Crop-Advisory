import sys
sys.path.append('engine')

from recommendation_engine import compute_final_score
import numpy as np

# Test case: ML None, should fallback to rule=70
rule_score = 70
ml_score = None
soil_score = 80
rotation_score = 60

final, contrib = compute_final_score(rule_score, ml_score, soil_score, rotation_score)
print(f"Final score: {final}")
print(f"Contributions: {contrib}")
print("ML used rule fallback:", contrib['ml_contribution'] == 0.4 * 70)
