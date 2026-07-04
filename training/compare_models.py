"""Train multiple regressors on the climate suitability dataset and compare metrics."""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train_model import (
    load_crops,
    build_vocab,
    build_crop_catalog,
    sample_environments,
    compute_targets,
    build_feature_matrix,
    aggregate_metrics,
)
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
OUTPUT_DIR = os.path.join(REPORT_DIR, 'legacy_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_dataset(samples: int, random_state: int, crops_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, list]]:
    crops = load_crops(crops_path)
    soils, seasons = build_vocab(crops)
    crop_catalog = build_crop_catalog(crops)
    if not crop_catalog:
        raise ValueError('No valid crop records found in dataset.')

    env_df = sample_environments(samples, soils, seasons, seed=random_state)
    y_df, _ = compute_targets(env_df, crop_catalog)
    X_df, metadata = build_feature_matrix(env_df, soils, seasons)
    return X_df.values, y_df.values, metadata


def make_models(random_state: int) -> Dict[str, MultiOutputRegressor]:
    models: Dict[str, MultiOutputRegressor] = {}

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    models['RandomForest'] = MultiOutputRegressor(rf, n_jobs=-1)

    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='reg:squarederror',
        n_jobs=0,
        random_state=random_state,
    )
    models['XGBoost'] = MultiOutputRegressor(xgb, n_jobs=-1)

    et = ExtraTreesRegressor(
        n_estimators=600,
        random_state=random_state,
        n_jobs=-1,
    )
    models['ExtraTrees'] = MultiOutputRegressor(et, n_jobs=-1)

    stack = StackingRegressor(
        estimators=[
            (
                'rf',
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            (
                'xgb',
                XGBRegressor(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective='reg:squarederror',
                    n_jobs=0,
                    random_state=random_state,
                ),
            ),
        ],
        final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0)),
        passthrough=True,
        n_jobs=-1,
    )
    models['Stacked RF+XGB'] = MultiOutputRegressor(stack, n_jobs=-1)

    return models


def plot_metrics(results: pd.DataFrame, output_path: str) -> None:
    metrics = ['mae', 'rmse', 'r2', 'accuracy']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, len(results.index)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(results.index, results[metric], color=colors)
        ax.set_title(metric.upper())
        ax.set_xticklabels(results.index, rotation=25, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.suptitle('Climate Suitability Model Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare multiple regressors on the climate suitability dataset.')
    parser.add_argument('--crops', default=os.path.join('data', 'crops_csv.csv'), help='Path to crops data (JSON/CSV).')
    parser.add_argument('--samples', type=int, default=8000, help='Number of synthetic environments to generate.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction (0-1).')
    parser.add_argument('--accuracy-tolerance', type=float, default=10.0, help='Tolerance for accuracy (% within +/- tolerance).')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    X, y, _ = build_dataset(args.samples, args.random_state, args.crops)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    models = make_models(args.random_state)
    records = []

    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        metrics = aggregate_metrics(y_test, preds, args.accuracy_tolerance)
        records.append({'model': name, **metrics})

    results_df = pd.DataFrame(records).set_index('model').sort_values('mae')

    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as fh:
        json.dump(records, fh, indent=2)

    plot_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plot_metrics(results_df, plot_path)

    print(f'Saved metrics to {metrics_path}')
    print(f'Saved comparison chart to {plot_path}')


if __name__ == '__main__':
    main()
