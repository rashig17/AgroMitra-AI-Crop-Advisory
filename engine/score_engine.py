"""
Rule-Based Scoring Engine (UPGRADED - Adaptive & Research-Level)

Features:
- Adaptive scoring based on deviation magnitude
- Anomaly detection integration
- Weight-based decision logic
- Research-level explainability

Outputs:
- rule_score (0–100)
- soil_score (0–100)
- penalties (for explainability with weights)
"""

from typing import Dict

# Weight-based scoring for structured decision logic
WEIGHTS = {
    "temp": 0.2,
    "rain": 0.2,
    "soil": 0.3,
    "season": 0.2
}


def score_crop(crop: Dict, user_env: Dict) -> Dict:
    """
    Adaptive rule-based crop scoring with anomaly detection and weights
    """
    score = 100
    penalties = []

    temp = user_env.get("temp")
    rain = user_env.get("rainfall")
    soil_type = user_env.get("soil_type") or user_env.get("Soil_Type")
    season = user_env.get("season")

    # =========================
    # 🔹 TEMPERATURE CHECK (Adaptive)
    # =========================
    if temp is not None:
        tmin = crop.get("Temp_Min")
        tmax = crop.get("Temp_Max")

        if tmin is not None and tmax is not None:
            ideal_temp = (tmin + tmax) / 2
            temp_diff = abs(temp - ideal_temp)

            if not (tmin <= temp <= tmax):
                # Adaptive penalty based on deviation
                penalty = min(20, temp_diff * 0.5)
                penalty_weighted = penalty * WEIGHTS["temp"]
                score -= penalty_weighted
                penalties.append({
                    "reason": f"Temperature mismatch",
                    "penalty": -round(penalty_weighted, 2),
                    "weight": WEIGHTS["temp"],
                    "details": f"Current: {temp}°C, Ideal: {ideal_temp}°C"
                })

    # =========================
    # 🔹 RAINFALL CHECK (Adaptive)
    # =========================
    if rain is not None:
        rmin = crop.get("Rain_Min")
        rmax = crop.get("Rain_Max")

        if rmin is not None and rmax is not None:
            ideal_rain = (rmin + rmax) / 2
            rain_diff = abs(rain - ideal_rain)

            if not (rmin <= rain <= rmax):
                # Adaptive penalty based on deviation
                penalty = min(20, rain_diff * 0.1)
                penalty_weighted = penalty * WEIGHTS["rain"]
                score -= penalty_weighted
                penalties.append({
                    "reason": f"Rainfall mismatch",
                    "penalty": -round(penalty_weighted, 2),
                    "weight": WEIGHTS["rain"],
                    "details": f"Current: {rain}mm, Ideal: {ideal_rain}mm"
                })

    # =========================
    # 🔹 ANOMALY DETECTION
    # =========================
    if user_env.get("temp_anomalies"):
        temp_anom_penalty = 10 * WEIGHTS["temp"]
        score -= temp_anom_penalty
        penalties.append({
            "reason": "Temperature anomaly",
            "penalty": -round(temp_anom_penalty, 2),
            "weight": WEIGHTS["temp"],
            "details": "Unusual temperature variation detected"
        })

    if user_env.get("rain_anomalies"):
        rain_anom_penalty = 10 * WEIGHTS["rain"]
        score -= rain_anom_penalty
        penalties.append({
            "reason": "Rainfall anomaly",
            "penalty": -round(rain_anom_penalty, 2),
            "weight": WEIGHTS["rain"],
            "details": "Unusual rainfall variation detected"
        })

    # =========================
    # 🔹 SOIL TYPE CHECK
    # =========================
    crop_soils = crop.get("Soil_Type", [])
    if soil_type and crop_soils:
        if soil_type not in crop_soils:
            soil_penalty = 40 * WEIGHTS["soil"]
            score -= soil_penalty
            penalties.append({
                "reason": "Soil type mismatch",
                "penalty": -round(soil_penalty, 2),
                "weight": WEIGHTS["soil"],
                "details": f"Current: {soil_type}, Required: {', '.join(crop_soils)}"
            })

    # =========================
    # 🔹 SEASON CHECK
    # =========================
    crop_seasons = crop.get("Season", [])
    if season and crop_seasons:
        if season not in crop_seasons:
            season_penalty = 30 * WEIGHTS["season"]
            score -= season_penalty
            penalties.append({
                "reason": "Season mismatch",
                "penalty": -round(season_penalty, 2),
                "weight": WEIGHTS["season"],
                "details": f"Current: {season}, Suitable: {', '.join(crop_seasons)}"
            })

    # =========================
    # 🔹 FINAL RULE SCORE
    # =========================
    rule_score = max(0, min(100, score))

    # =========================
    # 🔹 SOIL HEALTH SCORE
    # =========================
    soil_score = calculate_soil_score(user_env)

    return {
        "rule_score": rule_score,
        "soil_score": soil_score,
        "penalties": penalties
    }


# =========================
# 🔹 SOIL HEALTH FUNCTION
# =========================
def calculate_soil_score(user_env: Dict) -> float:
    """
    Simple soil health scoring using pH + organic matter
    """

    score = 50  # base

    ph = user_env.get("pH")
    om = user_env.get("organic_matter")

    # pH scoring
    if ph is not None:
        if 6.0 <= ph <= 7.5:
            score += 20
        elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
            score += 10
        else:
            score -= 15

    # Organic matter scoring
    if om is not None:
        if om >= 3:
            score += 20
        elif om >= 2:
            score += 10
        else:
            score -= 10

    return max(0, min(100, score))
