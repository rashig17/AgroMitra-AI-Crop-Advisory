import sys
import pandas as pd
sys.path.append('.')

from engine.recommendation_engine import recommend_top_k

# Mock data for test
mock_env = {'Temp': 25, 'Rain': 800, 'Soil_Type': 'Loam', 'Season': 'Kharif'}
mock_df = pd.DataFrame([{'Crop': 'Rice'}, {'Crop': 'Wheat'}, {'Crop': 'Maize'}])

# Mock engines returning fixed scores
def mock_rule(crop, env): return {'rule_score': 70 if crop['Crop']=='Rice' else 60, 'soil_score': 80}
def mock_rotation(crop, last): return 65

top_k = recommend_top_k(mock_env, mock_df, k=3, rule_engine=mock_rule, rotation_engine=mock_rotation)

print("Top K with new confidence:")
for rec in top_k:
    print(f"Crop: {rec['crop']}, Score: {rec['final_score']}, Confidence: {rec['confidence']}, Rank: {rec['rank']}")

