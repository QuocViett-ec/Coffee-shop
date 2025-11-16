"""
Kiểm tra xem các ML models có dự đoán giống nhau không
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.ml_models import MLForecaster
from models.train_test_split import get_X_y_split

print("="*70)
print(" KIỂM TRA PREDICTIONS CỦA CÁC MODELS")
print("="*70)

# Load data
X = pd.read_csv('data/processed/X.csv', index_col='date', parse_dates=True)
y = pd.read_csv('data/processed/y.csv', index_col='date', parse_dates=True).squeeze()

# Split
X_train, X_val, X_test, y_train, y_val, y_test = get_X_y_split(
    X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)

# Load trained models
print("\n[1] Loading trained models...")
xgb = MLForecaster('xgboost')
xgb.load_model('models/xgboost_model.pkl')

lgb = MLForecaster('lightgbm')
lgb.load_model('models/lightgbm_model.pkl')

rf = MLForecaster('random_forest')
rf.load_model('models/random_forest_model.pkl')

# Make predictions
print("\n[2] Making predictions on test set...")
y_pred_xgb = xgb.predict(X_test)
y_pred_lgb = lgb.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Compare predictions
print("\n" + "="*70)
print(" DETAILED PREDICTIONS COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Date': y_test.index.strftime('%Y-%m-%d'),
    'Actual': y_test.values,
    'XGBoost': y_pred_xgb,
    'LightGBM': y_pred_lgb,
    'RandomForest': y_pred_rf
})

print("\n", comparison_df.to_string(index=False))

# Calculate differences between models
print("\n" + "="*70)
print(" DIFFERENCES BETWEEN MODELS")
print("="*70)

diff_xgb_lgb = np.abs(y_pred_xgb - y_pred_lgb)
diff_xgb_rf = np.abs(y_pred_xgb - y_pred_rf)
diff_lgb_rf = np.abs(y_pred_lgb - y_pred_rf)

print(f"\nMean absolute difference:")
print(f"  XGBoost vs LightGBM:     ${diff_xgb_lgb.mean():.2f}")
print(f"  XGBoost vs RandomForest: ${diff_xgb_rf.mean():.2f}")
print(f"  LightGBM vs RandomForest: ${diff_lgb_rf.mean():.2f}")

print(f"\nMax absolute difference:")
print(f"  XGBoost vs LightGBM:     ${diff_xgb_lgb.max():.2f}")
print(f"  XGBoost vs RandomForest: ${diff_xgb_rf.max():.2f}")
print(f"  LightGBM vs RandomForest: ${diff_lgb_rf.max():.2f}")

# Check correlation between predictions
from scipy.stats import pearsonr

corr_xgb_lgb = pearsonr(y_pred_xgb, y_pred_lgb)[0]
corr_xgb_rf = pearsonr(y_pred_xgb, y_pred_rf)[0]
corr_lgb_rf = pearsonr(y_pred_lgb, y_pred_rf)[0]

print(f"\nCorrelation between predictions:")
print(f"  XGBoost vs LightGBM:     {corr_xgb_lgb:.4f}")
print(f"  XGBoost vs RandomForest: {corr_xgb_rf:.4f}")
print(f"  LightGBM vs RandomForest: {corr_lgb_rf:.4f}")

if corr_xgb_lgb > 0.99 and corr_xgb_rf > 0.99:
    print("\n⚠️  WARNING: Predictions are TOO SIMILAR!")
    print("    Correlation > 0.99 indicates models might be predicting the same thing")
else:
    print("\n✓ Predictions show reasonable diversity")

# Visualize predictions - DETAILED
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: All predictions
axes[0].plot(y_test.index, y_test.values, 'ko-', linewidth=3, markersize=6, label='Actual', zorder=3)
axes[0].plot(y_test.index, y_pred_xgb, 'r--', linewidth=2, marker='s', markersize=4, label='XGBoost', alpha=0.7)
axes[0].plot(y_test.index, y_pred_lgb, 'g--', linewidth=2, marker='^', markersize=4, label='LightGBM', alpha=0.7)
axes[0].plot(y_test.index, y_pred_rf, 'b--', linewidth=2, marker='v', markersize=4, label='RandomForest', alpha=0.7)
axes[0].set_ylabel('Revenue ($)', fontsize=11)
axes[0].set_title('Model Predictions Comparison', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Errors
errors_xgb = y_pred_xgb - y_test.values
errors_lgb = y_pred_lgb - y_test.values
errors_rf = y_pred_rf - y_test.values

axes[1].plot(y_test.index, errors_xgb, 'r-', linewidth=2, marker='s', markersize=4, label='XGBoost error', alpha=0.7)
axes[1].plot(y_test.index, errors_lgb, 'g-', linewidth=2, marker='^', markersize=4, label='LightGBM error', alpha=0.7)
axes[1].plot(y_test.index, errors_rf, 'b-', linewidth=2, marker='v', markersize=4, label='RandomForest error', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Prediction Error ($)', fontsize=11)
axes[1].set_title('Prediction Errors Over Time', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: Differences between models
axes[2].plot(y_test.index, diff_xgb_lgb, 'purple', linewidth=2, marker='o', label='|XGB - LGB|', alpha=0.7)
axes[2].plot(y_test.index, diff_xgb_rf, 'orange', linewidth=2, marker='s', label='|XGB - RF|', alpha=0.7)
axes[2].plot(y_test.index, diff_lgb_rf, 'brown', linewidth=2, marker='^', label='|LGB - RF|', alpha=0.7)
axes[2].set_xlabel('Date', fontsize=11)
axes[2].set_ylabel('Absolute Difference ($)', fontsize=11)
axes[2].set_title('Differences Between Model Predictions', fontsize=13, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_predictions_check.png', dpi=300, bbox_inches='tight')
print("\n✓ Detailed comparison plot saved to results/model_predictions_check.png")
plt.show()

# Statistical test for similarity
print("\n" + "="*70)
print(" DIAGNOSTIC: Are Models Too Similar?")
print("="*70)

# Check if models are just predicting mean/trend
pred_std_xgb = np.std(y_pred_xgb)
pred_std_lgb = np.std(y_pred_lgb)
pred_std_rf = np.std(y_pred_rf)
actual_std = np.std(y_test.values)

print(f"\nStandard deviation of predictions:")
print(f"  Actual test data:  ${actual_std:.2f}")
print(f"  XGBoost:           ${pred_std_xgb:.2f} ({pred_std_xgb/actual_std*100:.1f}% of actual)")
print(f"  LightGBM:          ${pred_std_lgb:.2f} ({pred_std_lgb/actual_std*100:.1f}% of actual)")
print(f"  RandomForest:      ${pred_std_rf:.2f} ({pred_std_rf/actual_std*100:.1f}% of actual)")

if pred_std_xgb/actual_std < 0.3 or pred_std_lgb/actual_std < 0.3 or pred_std_rf/actual_std < 0.3:
    print("\n⚠️  WARNING: Models not capturing enough variation!")
    print("    Predictions have much lower variance than actual data")
    print("    This suggests models are predicting conservatively (close to mean)")
else:
    print("\n✓ Models capturing reasonable variation")

# Check feature usage
print("\n" + "="*70)
print(" TOP FEATURES USED BY EACH MODEL")
print("="*70)

print("\nXGBoost top 5:")
xgb_importance = xgb.get_feature_importance(top_n=5)
for i, (feat, imp) in enumerate(xgb_importance.items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print("\nLightGBM top 5:")
lgb_importance = lgb.get_feature_importance(top_n=5)
for i, (feat, imp) in enumerate(lgb_importance.items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print("\nRandomForest top 5:")
rf_importance = rf.get_feature_importance(top_n=5)
for i, (feat, imp) in enumerate(rf_importance.items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

# Check if all using same features
common_features = set(xgb_importance.index[:5]) & set(lgb_importance.index[:5]) & set(rf_importance.index[:5])
if len(common_features) >= 3:
    print(f"\n⚠️  {len(common_features)} features appear in top 5 of ALL models: {common_features}")
    print("    This might explain why predictions are similar")

print("\n" + "="*70)
print(" CONCLUSION")
print("="*70)

if corr_xgb_lgb > 0.95 and pred_std_xgb/actual_std < 0.5:
    print("\n❌ PROBLEM DETECTED:")
    print("   - Models have very high correlation (> 0.95)")
    print("   - Models not capturing enough variation")
    print("   - All models likely overfitting to same features")
    print("\n   POSSIBLE CAUSES:")
    print("   1. Models overfitting to training data")
    print("   2. Features dominated by lag features (too predictive)")
    print("   3. Need more diverse features or regularization")
else:
    print("\n✓ Models show reasonable diversity")
    print("   Predictions are different enough to be useful")

print("\n✓ Diagnostic complete!")
