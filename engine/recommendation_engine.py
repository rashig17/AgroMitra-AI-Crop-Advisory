"""
Hybrid Recommendation Engine (UPGRADED)

Combines:
- Rule-based score (30%)
- ML score (40%)
- Soil health score (20%)
- Crop rotation score (10%)
- Anomaly penalties

Final Output:
- Ranked crop recommendations with explainability
- Final hybrid score with contribution breakdown
- Confidence score
"""

from typing import List, Dict, Optional
import numpy as np

# ==============================
# 🔹 CONFIGURABLE WEIGHTS
# ==============================
WEIGHTS = {
    "ml": 0.4,
    "rule": 0.3,
    "soil": 0.2,
    "rotation": 0.1
}


# ==============================
# 🔹 HYBRID SCORE FUNCTION
# ==============================
def compute_final_score(rule_score, ml_score, soil_score, rotation_score, user_env=None):
    """
    Compute final hybrid score using weighted sum
    All inputs expected in range 0–100
    
    Returns: final_score, contribution_breakdown
    """

    # Normalize safely
    rule = np.clip(rule_score if rule_score is not None else 50, 0, 100)
    ml = np.clip(ml_score if ml_score is not None else rule, 0, 100)
    soil = np.clip(soil_score if soil_score is not None else 50, 0, 100)
    rotation = np.clip(rotation_score if rotation_score is not None else 50, 0, 100)

    # Weighted hybrid formula using configurable weights
    final_score_raw = (
        WEIGHTS["ml"] * ml +
        WEIGHTS["rule"] * rule +
        WEIGHTS["soil"] * soil +
        WEIGHTS["rotation"] * rotation
    )

    # ==============================
    # 🔹 ANOMALY PENALTIES
    # ==============================
    anomaly_penalty = 0
    if user_env and user_env.get("temp_anomaly"):
        anomaly_penalty += 5
    if user_env and user_env.get("rain_anomaly"):
        anomaly_penalty += 5
    
    final_score = final_score_raw - anomaly_penalty

    # Contribution breakdown for explainability
    contributions = {
        "ml_contribution": round(WEIGHTS["ml"] * ml, 2),
        "rule_contribution": round(WEIGHTS["rule"] * rule, 2),
        "soil_contribution": round(WEIGHTS["soil"] * soil, 2),
        "rotation_contribution": round(WEIGHTS["rotation"] * rotation, 2),
        "anomaly_penalty": -anomaly_penalty
    }

    return round(np.clip(final_score, 0, 100), 2), contributions


# ==============================
# 🔹 CONFIDENCE SCORE
# ==============================
def compute_confidence(reliability: str, score_variance: float):
    """
    Compute confidence score based on:
    - data reliability
    - score stability
    """

    base = 0.8 if reliability == "high" else 0.6

    # Penalize high variance
    penalty = min(score_variance / 100, 0.3)

    confidence = (base - penalty) * 100
    return round(max(confidence, 10), 2)


# ==============================
# 🔹 MAIN RECOMMENDATION FUNCTION
# ==============================
def recommend_top_k(
    user_env: Dict,
    crops_df,
    last_crop: Optional[str] = None,
    k: int = 5,
    pipeline=None,
    ml_scores: Optional[Dict] = None,
    rule_engine=None,
    rotation_engine=None
) -> List[Dict]:
    """
    Generate top-K crop recommendations using hybrid scoring
    """

    recommendations = []

    for _, crop_row in crops_df.iterrows():
        crop = crop_row.to_dict()
        crop_name = crop.get("Crop")

        # ==============================
        # 🔹 RULE SCORE
        # ==============================
        if rule_engine:
            rule_result = rule_engine(crop, user_env)
            rule_score = rule_result.get("rule_score", 50)
            soil_score = rule_result.get("soil_score", 50)
        else:
            rule_score = 50
            soil_score = 50

        # ==============================
        # 🔹 ML SCORE
        # ==============================
        ml_score = None
        if ml_scores and crop_name in ml_scores:
            ml_score = ml_scores[crop_name]

        # ==============================
        # 🔹 ROTATION SCORE
        # ==============================
        if rotation_engine:
            try:
                rotation_score = rotation_engine(crop, last_crop, crops_df)
            except Exception as exc:
                print(f"  ⚠️ Rotation scoring failed for {crop_name}: {exc}")
                rotation_score = 50
        else:
            rotation_score = 50

        # ==============================
        # 🔹 FINAL HYBRID SCORE
        # ==============================
        final_score, contributions = compute_final_score(
            rule_score,
            ml_score,
            soil_score,
            rotation_score,
            user_env=user_env
        )

        print(f"  [Crop] {crop_name} | Rule: {rule_score:.1f} | ML: {f'{ml_score:.1f}' if ml_score is not None else '—'} | Final: {final_score:.1f}")
        recommendations.append({
            "crop": crop_name,
            "final_score": final_score,
            "rule_score": rule_score,
"ml_score": ml_score if ml_score is not None else None,
            "soil_score": soil_score,
            "rotation_score": rotation_score,
            # Explainability fields
            "ml_contribution": contributions.get("ml_contribution", 0),
            "rule_contribution": contributions.get("rule_contribution", 0),
            "soil_contribution": contributions.get("soil_contribution", 0),
            "rotation_contribution": contributions.get("rotation_contribution", 0),
            "anomaly_penalty": contributions.get("anomaly_penalty", 0)
        })


    # ==============================
    # 🔹 SORT RESULTS
    # ==============================
    recommendations = sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # ==============================
    # 🔹 COMPUTE CONFIDENCE
    # ==============================
    scores = [r["final_score"] for r in recommendations[:k]]
    variance = np.var(scores) if scores else 0

    reliability = user_env.get("reliability", "medium")

    confidence = compute_confidence(reliability, variance)

    # Attach ranking and per-crop relative confidence to top K
    top_k = recommendations[:k]
    if top_k:
        best_score = top_k[0]["final_score"]
    else:
        best_score = 0

    for i, rec in enumerate(top_k):
        rec["confidence"] = round(100 - abs(rec["final_score"] - best_score), 2)
        rec["rank"] = i + 1

    return top_k


# ==============================
# 🔹 SINGLE CROP SCORING (OPTIONAL)
# ==============================
def calculate_final_score(
    crop: Dict,
    user_env: Dict,
    last_crop: Optional[str],
    crops_df,
    pipeline=None,
    ml_scores: Optional[Dict] = None,
    rule_engine=None,
    rotation_engine=None
) -> Dict:
    """
    Compute score for a single crop (useful for testing/debugging)
    """

    crop_name = crop.get("Crop")

    # Rule score
    if rule_engine:
        rule_result = rule_engine(crop, user_env)
        rule_score = rule_result.get("rule_score", 50)
        soil_score = rule_result.get("soil_score", 50)
    else:
        rule_score = 50
        soil_score = 50

    # ML score
    ml_score = ml_scores.get(crop_name) if ml_scores else None

    # Rotation
    if rotation_engine:
        rotation_score = rotation_engine(crop_name, last_crop)
    else:
        rotation_score = 50

    # Final score
    final_score, _ = compute_final_score(
        rule_score,
        ml_score,
        soil_score,
        rotation_score
    )

    return {
        "crop": crop_name,
        "final_score": final_score,
        "rule_score": rule_score,
        "ml_score": ml_score,
        "soil_score": soil_score,
        "rotation_score": rotation_score
    }
