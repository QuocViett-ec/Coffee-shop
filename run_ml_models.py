"""
Train and Evaluate Machine Learning Models
XGBoost, LightGBM, Random Forest
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from models.ml_models import MLForecaster, compare_ml_models
from models.train_test_split import get_X_y_split


def plot_ml_forecasts(X_train, y_train, X_test, y_test, models, save_path='results/ml_forecasts.png'):
    """Plot ML model forecasts"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Full timeline
    axes[0].plot(y_train.index, y_train.values, label='Train', linewidth=2, color='blue', alpha=0.7)
    axes[0].plot(y_test.index, y_test.values, label='Test (Actual)', linewidth=3, color='black', marker='o', markersize=4)

    colors = ['red', 'green', 'orange']
    for i, (model_name, forecaster) in enumerate(models.items()):
        y_pred = forecaster.predict(X_test)
        axes[0].plot(y_test.index, y_pred, label=f'{model_name.upper()}',
                    linewidth=2, marker='s', markersize=3,
                    alpha=0.7, color=colors[i % len(colors)])

    axes[0].axvline(x=y_test.index[0], color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Revenue ($)', fontsize=11)
    axes[0].set_title('ML Model Forecasts', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)

    # Plot 2: Test period only (zoomed)
    axes[1].plot(y_test.index, y_test.values, label='Actual',
                linewidth=3, color='black', marker='o', markersize=5)

    for i, (model_name, forecaster) in enumerate(models.items()):
        y_pred = forecaster.predict(X_test)
        axes[1].plot(y_test.index, y_pred, label=f'{model_name.upper()}',
                    linewidth=2, marker='s', markersize=4,
                    alpha=0.7, color=colors[i % len(colors)])

    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Revenue ($)', fontsize=11)
    axes[1].set_title('Test Period Forecasts (Zoomed)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Forecast plot saved to {save_path}")


def plot_feature_importance(models, save_path='results/ml_feature_importance.png'):
    """Plot feature importance for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_names = list(models.keys())

    for i, (model_name, forecaster) in enumerate(models.items()):
        top_features = forecaster.get_feature_importance(top_n=15)

        axes[i].barh(range(len(top_features)), top_features.values, alpha=0.7)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features.index)
        axes[i].set_xlabel('Importance', fontsize=11)
        axes[i].set_title(f'{model_name.upper()} - Top 15 Features', fontsize=12, fontweight='bold')
        axes[i].invert_yaxis()
        axes[i].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to {save_path}")


def plot_ml_metrics_comparison(results, save_path='results/ml_metrics_comparison.png'):
    """Plot metrics comparison for ML models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAPE
    results_sorted = results.sort_values('MAPE')
    axes[0, 0].barh(results_sorted.index, results_sorted['MAPE'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('MAPE (%)', fontsize=11)
    axes[0, 0].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='x')
    axes[0, 0].axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Target < 15%')
    axes[0, 0].legend()

    # RMSE
    results_sorted = results.sort_values('RMSE')
    axes[0, 1].barh(results_sorted.index, results_sorted['RMSE'], color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('RMSE ($)', fontsize=11)
    axes[0, 1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='x')
    axes[0, 1].axvline(x=500, color='red', linestyle='--', alpha=0.5, label='Target < $500')
    axes[0, 1].legend()

    # MAE
    results_sorted = results.sort_values('MAE')
    axes[1, 0].barh(results_sorted.index, results_sorted['MAE'], color='seagreen', alpha=0.7)
    axes[1, 0].set_xlabel('MAE ($)', fontsize=11)
    axes[1, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')

    # R²
    results_sorted = results.sort_values('R2', ascending=True)
    axes[1, 1].barh(results_sorted.index, results_sorted['R2'], color='mediumpurple', alpha=0.7)
    axes[1, 1].set_xlabel('R² Score', fontsize=11)
    axes[1, 1].set_title('R-Squared Score', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    axes[1, 1].axvline(x=0.85, color='red', linestyle='--', alpha=0.5, label='Target > 0.85')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics plot saved to {save_path}")


def main():
    print("="*70)
    print(" MACHINE LEARNING MODELS TRAINING & EVALUATION")
    print("="*70)

    # Load data
    print("\n[1/5] Loading feature-engineered data...")
    X = pd.read_csv('data/processed/X.csv', index_col='date', parse_dates=True)
    y = pd.read_csv('data/processed/y.csv', index_col='date', parse_dates=True).squeeze()
    print(f"✓ Loaded: X shape {X.shape}, y shape {y.shape}")

    # Split data
    print("\n[2/5] Creating train/val/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_X_y_split(
        X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    # Train ML models
    print("\n[3/5] Training ML models (XGBoost, LightGBM, Random Forest)...")
    results, models = compare_ml_models(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save results
    print("\n[4/5] Saving results...")
    results.to_csv('results/ml_model_results.csv')
    print(f"✓ Results saved to results/ml_model_results.csv")

    # Save models
    for model_name, forecaster in models.items():
        model_path = f'models/{model_name}_model.pkl'
        forecaster.save_model(model_path)

    # Create visualizations
    print("\n[5/5] Creating visualizations...")
    plot_ml_forecasts(X_train, y_train, X_test, y_test, models)
    plot_feature_importance(models)
    plot_ml_metrics_comparison(results)

    # Save summary
    with open('results/ml_summary.txt', 'w') as f:
        f.write("MACHINE LEARNING MODELS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Date range: {X.index.min()} to {X.index.max()}\n\n")
        f.write("Model Performance:\n")
        f.write("-"*70 + "\n")
        f.write(results.to_string())
        f.write("\n\n")

        best_model = results['MAPE'].idxmin()
        f.write(f"🏆 Best ML Model: {best_model.upper()}\n")
        f.write(f"   MAPE: {results.loc[best_model, 'MAPE']:.2f}%\n")
        f.write(f"   RMSE: ${results.loc[best_model, 'RMSE']:.2f}\n")
        f.write(f"   MAE: ${results.loc[best_model, 'MAE']:.2f}\n")
        f.write(f"   R²: {results.loc[best_model, 'R2']:.4f}\n")

        # Feature importance
        f.write(f"\n\nTop 20 Features ({best_model.upper()}):\n")
        f.write("-"*70 + "\n")
        top_features = models[best_model].get_feature_importance(top_n=20)
        for i, (feat, imp) in enumerate(top_features.items(), 1):
            f.write(f"{i:2d}. {feat:40s}: {imp:.6f}\n")

    print(f"✓ Summary saved to results/ml_summary.txt")

    print("\n" + "="*70)
    print(" MACHINE LEARNING MODELS COMPLETE")
    print("="*70)

    return models, results


if __name__ == "__main__":
    models, results = main()
