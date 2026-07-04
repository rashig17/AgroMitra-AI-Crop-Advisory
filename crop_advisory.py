import json
import os

IMPACT_FILE = "data/crop_impacts.json"

def load_impact_data():
    """Load crop impact dataset safely with fallbacks."""
    try:
        if os.path.exists(IMPACT_FILE):
            with open(IMPACT_FILE, "r") as f:
                return json.load(f)
        else:
            print("⚠️ crop_impacts.json not found — using empty impact data")
            return {"crop_impacts": {}}
    except Exception as e:
        print(f"❌ Failed to load impact data: {e}")
        return {"crop_impacts": {}}


def calculate_soil_health_score(soil_data, impact_data):
    """Return soil health score (0–100) backed by real chemistry measurements."""

    score = 60  # lean optimistic baseline when data exists
    impacts = []

    def _record(factor: str, impact: int, note: str):
        if impact:
            impacts.append({"factor": factor, "impact": impact, "note": note})

    ph_value = soil_data.get("ph")
    if ph_value is not None:
        if 6.0 <= ph_value <= 7.5:
            score += 12
            _record("Soil pH", +12, f"pH {ph_value:.1f} is ideal for most crops")
        elif 5.5 <= ph_value < 6.0 or 7.5 < ph_value <= 8.0:
            score += 4
            _record("Soil pH", +4, f"pH {ph_value:.1f} is acceptable but worth monitoring")
        elif 5.0 <= ph_value < 5.5 or 8.0 < ph_value <= 8.5:
            score -= 10
            _record("Soil pH", -10, f"pH {ph_value:.1f} may limit nutrient availability")
        else:
            score -= 18
            _record("Soil pH", -18, f"pH {ph_value:.1f} is stressful for most crops")

    om_pct = soil_data.get("organic_matter_pct")
    if om_pct is None:
        om_str = soil_data.get("organic_matter")
        if isinstance(om_str, str) and om_str.endswith("%"):
            try:
                om_pct = float(om_str.strip("%"))
            except ValueError:
                om_pct = None

    if om_pct is not None:
        if om_pct >= 3.0:
            score += 10
            _record("Organic matter", +10, f"OM {om_pct:.1f}% sustains soil biology")
        elif 2.0 <= om_pct < 3.0:
            score += 4
            _record("Organic matter", +4, f"OM {om_pct:.1f}% is healthy")
        elif 1.2 <= om_pct < 2.0:
            score -= 6
            _record("Organic matter", -6, f"OM {om_pct:.1f}% is modest; add residues/compost")
        else:
            score -= 14
            _record("Organic matter", -14, f"OM {om_pct:.1f}% is low; boost organic inputs")

    nd = impact_data.get("nutrient_depletion", {})
    if nd:
        val = nd.get("impact", 0)
        score += val
        _record("nutrient depletion", val, "Crop rotation history adjustment")

    dr = impact_data.get("disease_risk", {})
    if dr:
        val = dr.get("impact", 0)
        score += val
        _record("disease risk", val, "Rotation disease pressure adjustment")

    pd = impact_data.get("physical_degradation", {})
    if pd:
        val = pd.get("impact", 0)
        score += val
        _record("physical degradation", val, "Tillage/compaction adjustment")

    score = max(0, min(100, score))

    if score >= 80:
        status = "Healthy"
    elif score >= 55:
        status = "Needs Attention"
    else:
        status = "At Risk"

    return {
        "score": score,
        "risk_level": status,
        "recommendation": status,
        "impacts": impacts
    }


def calculate_rainfall_impact(rainfall, crop_name, impact_data):
    """Score rainfall suitability for the crop."""
    crop_imp = impact_data.get("crop_impacts", {}).get(crop_name, {})
    rain = crop_imp.get("rainfall_sensitivity", {"impact": 0})

    base = 60  # neutral baseline
    rain_penalty = rain.get("impact", 0)

    final = max(0, min(100, base + rain_penalty))

    if final >= 75:
        rec = "Highly Recommended"
    elif final >= 50:
        rec = "Recommended with Caution"
    else:
        rec = "Not Recommended"

    return {
        "score": final,
        "severity": "Moderate" if rain_penalty else "Low",
        "penalty": -rain_penalty,
        "risk_level": rec,
        "recommendation": rec
    }
