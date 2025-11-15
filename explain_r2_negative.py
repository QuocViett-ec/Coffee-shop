"""
Demo: Tại sao R² âm trong Time Series Forecasting
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tạo data giống coffee shop (có strong upward trend)
dates = pd.date_range('2023-01-01', periods=20, freq='D')
# Revenue tăng mạnh theo thời gian (giống +124% growth)
actual_revenue = np.array([
    3000, 3100, 3200, 3300, 3400,  # Tuần 1
    3500, 3600, 3700, 3800, 3900,  # Tuần 2
    4000, 4100, 4200, 4300, 4400,  # Tuần 3
    4500, 4600, 4700, 4800, 4900   # Tuần 4 (test set)
])

# Train/test split
train_actual = actual_revenue[:15]  # 15 ngày đầu
test_actual = actual_revenue[15:]   # 5 ngày cuối

print("="*70)
print(" TẠI SAO R² ÂM TRONG TIME SERIES?")
print("="*70)

# Baseline: Predict bằng TRUNG BÌNH training set
baseline_mean = train_actual.mean()
baseline_predictions = np.array([baseline_mean] * len(test_actual))

print(f"\n1. BASELINE (predict bằng trung bình train):")
print(f"   Training mean: ${baseline_mean:,.2f}")
print(f"   Predictions: Tất cả = ${baseline_mean:,.2f}")

# Model predictions (ví dụ model không tốt, overfit)
# Giả sử model predict thấp hơn actual
model_predictions = test_actual - 300  # Model systematically underpredict

print(f"\n2. MODEL PREDICTIONS:")
for i, (actual, pred) in enumerate(zip(test_actual, model_predictions)):
    print(f"   Day {i+1}: Actual ${actual:,} | Predicted ${pred:,} | Error ${actual-pred:,}")

# Calculate R²
from sklearn.metrics import r2_score, mean_squared_error

r2_model = r2_score(test_actual, model_predictions)
r2_baseline = r2_score(test_actual, baseline_predictions)

mse_model = mean_squared_error(test_actual, model_predictions)
mse_baseline = mean_squared_error(test_actual, baseline_predictions)

print(f"\n{'='*70}")
print(" KẾT QUẢ:")
print(f"{'='*70}")
print(f"\nMODEL:")
print(f"  MSE:  {mse_model:,.2f}")
print(f"  R²:   {r2_model:.4f}  {'← ÂM!' if r2_model < 0 else ''}")

print(f"\nBASELINE (trung bình):")
print(f"  MSE:  {mse_baseline:,.2f}")
print(f"  R²:   {r2_baseline:.4f}")

print(f"\n{'='*70}")
print(" GIẢI THÍCH:")
print(f"{'='*70}")
print(f"""
R² = 1 - (MSE_model / MSE_baseline)
R² = 1 - ({mse_model:,.2f} / {mse_baseline:,.2f})
R² = 1 - {mse_model/mse_baseline:.4f}
R² = {r2_model:.4f}

➡️ R² ÂM nghĩa là: MSE_model > MSE_baseline
➡️ Model dự đoán TỆ HƠN dự đoán bằng trung bình!
""")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Predictions comparison
x = range(len(test_actual))
axes[0].plot(x, test_actual, 'ko-', linewidth=3, markersize=8, label='Actual', zorder=3)
axes[0].plot(x, baseline_predictions, 'b--', linewidth=2, marker='s', markersize=6,
             label=f'Baseline (mean=${baseline_mean:.0f})', alpha=0.7)
axes[0].plot(x, model_predictions, 'r--', linewidth=2, marker='^', markersize=6,
             label='Model Predictions', alpha=0.7)
axes[0].set_xlabel('Test Day', fontsize=11)
axes[0].set_ylabel('Revenue ($)', fontsize=11)
axes[0].set_title('Why R² is Negative', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Errors comparison
baseline_errors = np.abs(test_actual - baseline_predictions)
model_errors = np.abs(test_actual - model_predictions)

axes[1].bar(x, baseline_errors, alpha=0.6, label='Baseline Errors', color='blue')
axes[1].bar(x, model_errors, alpha=0.6, label='Model Errors', color='red')
axes[1].set_xlabel('Test Day', fontsize=11)
axes[1].set_ylabel('Absolute Error ($)', fontsize=11)
axes[1].set_title('Error Comparison', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('r2_negative_explanation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved to r2_negative_explanation.png")

# Coffee shop case
print(f"\n{'='*70}")
print(" ÁP DỤNG VÀO COFFEE SHOP PROJECT:")
print(f"{'='*70}")
print("""
Coffee shop revenue có STRONG UPWARD TREND (+124% growth):
- Jan: ~$2,400/day
- Jun: ~$5,400/day

Test set: Những ngày cuối Jun với revenue cao (~$5,500-6,400)
Training mean: ~$3,860

Nếu predict bằng training mean ($3,860):
→ Sai rất nhiều! Vì Jun cao hơn nhiều

Model predictions cũng sai nhưng KHÔNG SÁI BẰNG baseline
→ R² vẫn âm nhưng model VẪN TỐT HƠN baseline!

📊 Chú ý metrics quan trọng hơn:
   - MAPE: 6.68% (MA_3) ← Đây là metric tốt!
   - RMSE: $468 (MA_3) ← Đây cũng tốt!
   - R² âm: Không sao, vì baseline (mean) quá tệ với trending data
""")

print(f"\n{'='*70}")
print(" KẾT LUẬN:")
print(f"{'='*70}")
print("""
✓ R² ÂM KHÔNG có nghĩa là model TỆ!
✓ Nó chỉ nghĩa là model tệ hơn "predict bằng trung bình"
✓ Với time series có trend mạnh, "predict bằng trung bình" là baseline TỆ
✓ → R² không phải metric tốt cho time series có trend!

📌 Nên dùng metrics này thay thế:
   1. MAPE (Mean Absolute Percentage Error) ← BEST
   2. RMSE (Root Mean Squared Error)
   3. MAE (Mean Absolute Error)
   4. MBD (Mean Bias Deviation) - check systematic error
""")
