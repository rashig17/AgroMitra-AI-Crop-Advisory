"""Crop rotation scoring engine (UPGRADED - Normalized to 0-100 scale).

Evaluates rotation score based on crop compatibility after previous crop.
- Assigns higher scores if crop rotations help mitigate counters from last crop
- Penalizes same-crop repetition
- Normalized to 0-100 scale for hybrid system alignment
"""

import os
import pandas as pd


def get_rotation_score(crop_row: dict, last_crop: str = None, crops_df: pd.DataFrame = None) -> float:
    """Calculate rotation score (0..100) for a crop given the last crop grown.

    Args:
        crop_row: Dict with crop properties including 'Crop' and 'Rotation_Benefit'.
        last_crop: Name of crop grown in previous season (None if no prior crop).
        crops_df: DataFrame with all crops (for lookup). If None, loads from data/crops_csv.csv.

    Returns:
        Float score from 0 to 100 representing rotation benefit, normalized for hybrid system.
    """
    # Default score if no prior crop
    if last_crop is None:
        return 50.0  # Neutral score (middle of 0-100 scale)

    # Load crops data if not provided
    if crops_df is None:
        crops_path = os.path.join('data', 'crops_csv.csv')
        if os.path.exists(crops_path):
            crops_df = pd.read_csv(crops_path)
        else:
            return 50.0  # Default if crops data unavailable

    # ============================================
    # 🔹 PENALTY: Avoid same crop repetition
    # ============================================
    current_crop = str(crop_row.get('Crop', '')).lower()
    if current_crop == last_crop.lower():
        print(f"⚠️ Rotation penalty: {last_crop} → {current_crop} (same crop)")
        return 0.0  # No rotation benefit for same crop

    # Find last crop row
    last_crop_rows = crops_df[crops_df['Crop'].str.lower() == last_crop.lower()]
    if last_crop_rows.empty:
        return 50.0  # Default if last crop not found

    last_crop_row = last_crop_rows.iloc[0]

    # Extract counters from last crop (e.g., "Nitrogen depletion;Pest buildup")
    last_crop_counters = str(last_crop_row.get('Counters', '')).lower()
    
    # Extract rotation benefit of current crop
    current_rotation_benefit = str(crop_row.get('Rotation_Benefit', '')).lower()

    # Calculate overlap between rotation benefit and last crop counters
    counter_list = [c.strip() for c in last_crop_counters.split(';') if c.strip()]
    benefit_list = [b.strip() for b in current_rotation_benefit.split(';') if b.strip()]

    # Count matches using substring matching
    matches = 0
    for benefit in benefit_list:
        for counter in counter_list:
            if benefit and counter and benefit in counter:
                matches += 1

    # ============================================
    # 🔹 STRUCTURED SCORING (0-100 scale)
    # ============================================
    # Score based on number of benefits that counter previous crop issues
    if matches == 0:
        rotation_score = 30  # Poor rotation
    elif matches <= 2:
        rotation_score = 60  # Moderate rotation
    elif matches <= 4:
        rotation_score = 80  # Good rotation
    else:
        rotation_score = 100  # Excellent rotation

    return float(rotation_score)
