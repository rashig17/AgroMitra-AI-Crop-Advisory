"""Train a multi-output regressor that maps environment conditions to crop scores.

The model consumes user-facing inputs (soil type, rainfall, temperature, pH, and
season) and predicts a suitability score (0-100) for every crop in the catalog.
Scores are derived directly from agronomic ranges in ``crops_csv.csv`` so the ML
model learns an intrinsic climate suitability surface rather than mimicking the
legacy rule engine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

RANDOM_STATE = 42
DEFAULT_SAMPLES = 8000
DEFAULT_SEASONS = ['Kharif', 'Rabi', 'Summer', 'Winter', 'Year-round']
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TEMPERATURE_WEIGHT = 0.4
RAINFALL_WEIGHT = 0.3
SOIL_WEIGHT = 0.1
PH_WEIGHT = 0.1
SEASON_WEIGHT = 0.1
WEIGHT_TOTAL = TEMPERATURE_WEIGHT + RAINFALL_WEIGHT + SO 

def load_crops(path: str) -> List[Dict]:
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError('crops JSON must contain a list of records')
        return data
    return pd.read_csv(path).to_dict(orient='records')


def _as_iterable(value: Iterable | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value)
    for sep in (';', ',', '|'):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    cleaned = text.strip()
    return [cleaned] if cleaned else []


def build_vocab(crops: Sequence[Dict]) -> Tuple[List[str], List[str]]:
    soil_set = set()
    season_set = set()
    for crop in crops:
        for soil in _as_iterable(crop.get('Soil_Type')):
            soil_set.add(soil.strip())
        for season in _as_iterable(crop.get('Season')):
            season_set.add(season.strip())
    if not soil_set:
        soil_set.update(['Loam', 'Clay', 'Sandy'])
    if not season_set:
        season_set.update(DEFAULT_SEASONS)
    soil_list = sorted(soil_set)
    season_list = sorted(
        season_set,
        key=lambda s: (s not in DEFAULT_SEASONS, DEFAULT_SEASONS.index(s) if s in DEFAULT_SEASONS else len(DEFAULT_SEASONS))
    )
    return soil_list, season_list


def _normalise_label(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned.lower().replace('/', '_').replace(' ', '_')


def _safe_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_crop_catalog(crops: Sequence[Dict]) -> List[Dict[str, object]]:
    catalog: List[Dict[str, object]] = []
    for crop in crops:
        name = crop.get('Crop') or crop.get('Crop_Name') or crop.get('Name')
        if not name:
            continue
        soils = {
            lbl
            for lbl in (_normalise_label(item) for item in _as_iterable(crop.get('Soil_Type')))
            if lbl
        }
        seasons = {
            lbl
            for lbl in (_normalise_label(item) for item in _as_iterable(crop.get('Season')))
            if lbl
        }
        catalog.append({
            'name': str(name),
            'temp_min': _safe_float(crop.get('Temp_Min')),
            'temp_max': _safe_float(crop.get('Temp_Max')),
            'rain_min': _safe_float(crop.get('Rain_Min')),
            'rain_max': _safe_float(crop.get('Rain_Max')),
            'ph_min': _safe_float(crop.get('pH_Min')),
            'ph_max': _safe_float(crop.get('pH_Max')),
            'soil_labels': soils,
            'season_labels': seasons,
        })
    return catalog


def _range_similarity(value: float, lower: float | None, upper: float | None) -> float:
    if lower is None and upper is None:
        return 50.0
    if lower is None:
        lower = upper
    if upper is None:
        upper = lower
    if lower is None or upper is None:
        return 50.0
    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        lower, upper = upper, lower
    width = max(upper - lower, 1.0)
    if lower <= value <= upper:
        return 100.0
    diff = lower - value if value < lower else value - upper
    score = max(0.0, 100.0 - (diff / width) * 100.0)
    return float(score)


def _climate_suitability(env_row: pd.Series, crop_profile: Dict[str, object]) -> float:
    temperature = float(env_row['temperature'])
    rainfall = float(env_row['rainfall'])
    ph_value = float(env_row['ph'])
    soil_label = _normalise_label(env_row['soil_type'])
    season_label = _normalise_label(env_row['season'])

    temp_score = _range_similarity(temperature, crop_profile.get('temp_min'), crop_profile.get('temp_max'))
    rain_score = _range_similarity(rainfall, crop_profile.get('rain_min'), crop_profile.get('rain_max'))

    soil_allowed = crop_profile.get('soil_labels') or set()
    season_allowed = crop_profile.get('season_labels') or set()

    soil_score = 100.0 if not soil_allowed or (soil_label and soil_label in soil_allowed) else 0.0
    ph_min = crop_profile.get('ph_min')
    ph_max = crop_profile.get('ph_max')
    if ph_min is None and ph_max is None:
        ph_score = 100.0
    else:
        ph_min = float(ph_min) if ph_min is not None else float('-inf')
        ph_max = float(ph_max) if ph_max is not None else float('inf')
        ph_score = 100.0 if ph_min <= ph_value <= ph_max else 0.0
    season_score = 100.0 if not season_allowed or (season_label and season_label in season_allowed) else 0.0

    weighted = (
        temp_score * TEMPERATURE_WEIGHT
        + rain_score * RAINFALL_WEIGHT
        + soil_score * SOIL_WEIGHT
        + ph_score * PH_WEIGHT
        + season_score * SEASON_WEIGHT
    )
    normalised = weighted / WEIGHT_TOTAL if WEIGHT_TOTAL else weighted
    return float(np.clip(normalised, 0.0, 100.0))


def sample_environments(n_samples: int, soils: Sequence[str], seasons: Sequence[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'soil_type': rng.choice(soils, size=n_samples, replace=True),
        'temperature': rng.uniform(12.0, 42.0, size=n_samples),
        'rainfall': rng.uniform(200.0, 2800.0, size=n_samples),
        'ph': rng.uniform(5.0, 8.2, size=n_samples),
        'season': rng.choice(seasons, size=n_samples, replace=True),
    })
    df['soil_type'] = df['soil_type'].astype(str).str.strip()
    df['season'] = df['season'].astype(str).str.strip()
    return df


def compute_targets(env_df: pd.DataFrame, crop_catalog: Sequence[Dict[str, object]]) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[List[float]] = []
    for _, env_row in env_df.iterrows():
        scores = [_climate_suitability(env_row, crop_profile) for crop_profile in crop_catalog]
        rows.append(scores)
    crop_names = [str(crop_profile['name']) for crop_profile in crop_catalog]
    y = pd.DataFrame(rows, columns=crop_names)
    return y, crop_names


def build_feature_matrix(env_df: pd.DataFrame, soils: Sequence[str], seasons: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = env_df.copy()
    soil_cols = [f'soil__{_normalise_label(s)}' for s in soils]
    season_cols = [f'season__{_normalise_label(s)}' for s in seasons]

    soil_dummies = pd.get_dummies(df['soil_type'].apply(_normalise_label), prefix='soil')
    season_dummies = pd.get_dummies(df['season'].apply(_normalise_label), prefix='season')

    for col in soil_cols:
        if col not in soil_dummies.columns:
            soil_dummies[col] = 0
    for col in season_cols:
        if col not in season_dummies.columns:
            season_dummies[col] = 0

    base = df[['rainfall', 'temperature', 'ph']].reset_index(drop=True)
    soil_dummies = soil_dummies[soil_cols].reset_index(drop=True)
    season_dummies = season_dummies[season_cols].reset_index(drop=True)

    X = pd.concat([base, soil_dummies, season_dummies], axis=1)
    metadata = {
        'feature_columns': list(X.columns),
        'soil_columns': soil_cols,
        'season_columns': season_cols,
        'base_features': ['rainfall', 'temperature', 'ph'],
    }
    return X, metadata


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100.0
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'accuracy': float(accuracy)
    }


def save_prepared(X: pd.DataFrame, y: pd.DataFrame, metadata: Dict[str, List[str]], crop_names: Sequence[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    X.to_csv(os.path.join(out_dir, 'X.csv'), index=False)
    y.to_csv(os.path.join(out_dir, 'y.csv'), index=False)
    payload = {**metadata, 'crop_names': list(crop_names)}
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)


def save_artifacts(
    model: object,
    metrics: Dict[str, Dict[str, float]],
    features: Sequence[str],
    crop_names: Sequence[str],
    out_dir: str,
    epochs: int,
    accuracy_tolerance: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, 'env_model.joblib'))
    joblib.dump(list(features), os.path.join(out_dir, 'features.joblib'))
    joblib.dump(list(crop_names), os.path.join(out_dir, 'crop_names.joblib'))
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'model': os.path.join(out_dir, 'env_model.joblib'),
        'features': os.path.join(out_dir, 'features.joblib'),
        'crop_names': os.path.join(out_dir, 'crop_names.joblib'),
        'validation': metrics.get('val'),
        'test': metrics['test'],
        'epochs': int(epochs),
        'accuracy_tolerance': float(accuracy_tolerance),
    }
    with open(os.path.join(out_dir, 'training_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)


def _make_scatter(actual: np.ndarray, predicted: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, s=8, alpha=0.4, color='#1f77b4')
    plt.plot([0, 100], [0, 100], color='#ff7f0e', linewidth=1.2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Predicted vs Actual Suitability Scores')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _make_residual_hist(residuals: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, color='#2ca02c', alpha=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _make_distribution_overlay(actual: np.ndarray, predicted: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(actual, bins=40, alpha=0.6, label='Actual', color='#1f77b4')
    plt.hist(predicted, bins=40, alpha=0.6, label='Predicted', color='#ff7f0e')
    plt.xlabel('Suitability Score')
    plt.ylabel('Count')
    plt.title('Score Distribution Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def generate_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    actual = y_true.flatten()
    predicted = y_pred.flatten()
    residuals = actual - predicted
    _make_scatter(actual, predicted, os.path.join(report_dir, 'pred_vs_actual.png'))
    _make_residual_hist(residuals, os.path.join(report_dir, 'residuals_hist.png'))
    _make_distribution_overlay(actual, predicted, os.path.join(report_dir, 'dist_comparison.png'))


def save_evaluation_report(test_metrics: Dict[str, float], report_path: str, epochs: int) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    plots = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as fh:
                existing = json.load(fh)
            if isinstance(existing, dict):
                plots = existing.get('plots', {}) or {}
        except Exception:
            plots = {}
    payload = {
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': {
            'MAE': float(test_metrics.get('mae', 0.0)),
            'RMSE': float(test_metrics.get('rmse', 0.0)),
            'R2': float(test_metrics.get('r2', 0.0)),
            'Accuracy': float(test_metrics.get('accuracy', 0.0)),
            'Epochs': int(epochs),
        },
        'plots': plots,
    }
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)


def main(args: argparse.Namespace) -> None:
    crops = load_crops(args.crops)
    soils, seasons = build_vocab(crops)
    crop_catalog = build_crop_catalog(crops)
    if not crop_catalog:
        raise ValueError('No valid crop records found in dataset.')

    env_df = sample_environments(args.samples, soils, seasons, seed=args.random_state)
    y_df, crop_names = compute_targets(env_df, crop_catalog)
    X_df, metadata = build_feature_matrix(env_df, soils, seasons)

    save_prepared(X_df, y_df, metadata, crop_names, args.prepared)

    X = X_df.values
    y = y_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    if args.val_size > 0.0:
        val_fraction = args.val_size / (1.0 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction, random_state=args.random_state
        )

    base_estimator = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base_estimator, n_jobs=-1)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    metrics: Dict[str, Dict[str, float] | None] = {}
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        metrics['val'] = aggregate_metrics(y_val, val_pred, args.accuracy_tolerance)
    else:
        metrics['val'] = None
    metrics['test'] = aggregate_metrics(y_test, test_pred, args.accuracy_tolerance)

    report_dir = os.path.join(PROJECT_ROOT, 'reports')
    generate_evaluation_plots(y_test, test_pred, report_dir)

    save_artifacts(
        model,
        metrics,
        metadata['feature_columns'],
        crop_names,
        args.out,
        args.epochs,
        args.accuracy_tolerance,
    )

    save_evaluation_report(
        metrics['test'],
        os.path.join(report_dir, 'evaluation_report.json'),
        args.epochs,
    )

    print('Training complete:')
    if metrics['val']:
        print(
            "  Validation -> MAE {mae:.2f}, RMSE {rmse:.2f}, R2 {r2:.3f}, ACC {acc:.2f}%".format(
                mae=metrics['val']['mae'],
                rmse=metrics['val']['rmse'],
                r2=metrics['val']['r2'],
                acc=metrics['val']['accuracy'],
            )
        )
    else:
        print("  Validation -> skipped (val_size=0.0)")
    print(
        "  Test       -> MAE {mae:.2f}, RMSE {rmse:.2f}, R2 {r2:.3f}, ACC {acc:.2f}%".format(
            mae=metrics['test']['mae'],
            rmse=metrics['test']['rmse'],
            r2=metrics['test']['r2'],
            acc=metrics['test']['accuracy'],
        )
    )
    print(f"  Epochs     -> {args.epochs}")
    print(f"  Accuracy tolerance (+/-) -> {args.accuracy_tolerance:.2f} points")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train environment-driven crop suitability model.')
    parser.add_argument('--crops', default=os.path.join('data', 'crops_csv.csv'), help='Path to crops data (JSON/CSV).')
    parser.add_argument('--prepared', default=os.path.join('data', 'prepared'), help='Directory to write prepared datasets.')
    parser.add_argument('--out', default='models', help='Directory to write model artifacts.')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES, help='Number of synthetic environments to generate.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (0-1).')
    parser.add_argument('--val-size', type=float, default=0.0, help='Validation fraction of the full dataset.')
    parser.add_argument('--n-estimators', type=int, default=500, help='RandomForest tree count.')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=1, help='Logical epoch count for logging purposes.')
    parser.add_argument('--accuracy-tolerance', type=float, default=10.0, help='Tolerance (in score points) for accuracy computation.')
    main(parser.parse_args())
