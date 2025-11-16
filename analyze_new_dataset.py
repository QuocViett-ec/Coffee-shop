"""Quick analysis of new Coffe_sales.csv dataset"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANALYZING NEW DATASET: Coffe_sales.csv")
print("="*80)

# Load new dataset
df_new = pd.read_csv('/home/user/Coffee-shop/Coffe_sales.csv')

print("\n1. DATASET OVERVIEW")
print(f"   Shape: {df_new.shape}")
print(f"   Columns: {df_new.columns.tolist()}")
print(f"\n   Data types:\n{df_new.dtypes}")
print(f"\n   Missing values:\n{df_new.isnull().sum()}")

# Convert Date column
df_new['Date'] = pd.to_datetime(df_new['Date'])

print("\n2. DATE RANGE")
print(f"   Start: {df_new['Date'].min()}")
print(f"   End: {df_new['Date'].max()}")
print(f"   Total days: {(df_new['Date'].max() - df_new['Date'].min()).days + 1}")
print(f"   Unique days: {df_new['Date'].nunique()}")

print("\n3. REVENUE ANALYSIS")
print(f"   Total revenue: ${df_new['money'].sum():,.2f}")
print(f"   Mean transaction: ${df_new['money'].mean():.2f}")
print(f"   Median transaction: ${df_new['money'].median():.2f}")
print(f"   Revenue range: ${df_new['money'].min():.2f} - ${df_new['money'].max():.2f}")

# Create daily revenue
daily_revenue = df_new.groupby('Date')['money'].sum().reset_index()
daily_revenue.columns = ['date', 'revenue']

print("\n4. DAILY REVENUE STATS")
print(f"   Days with data: {len(daily_revenue)}")
print(f"   Mean daily revenue: ${daily_revenue['revenue'].mean():,.2f}")
print(f"   Median daily revenue: ${daily_revenue['revenue'].median():,.2f}")
print(f"   Std daily revenue: ${daily_revenue['revenue'].std():,.2f}")
print(f"   Min daily revenue: ${daily_revenue['revenue'].min():,.2f}")
print(f"   Max daily revenue: ${daily_revenue['revenue'].max():,.2f}")

print("\n5. PRODUCT ANALYSIS")
print(f"   Unique coffee types: {df_new['coffee_name'].nunique()}")
print(f"\n   Top products by count:")
print(df_new['coffee_name'].value_counts().head(10))
print(f"\n   Top products by revenue:")
product_revenue = df_new.groupby('coffee_name')['money'].sum().sort_values(ascending=False)
print(product_revenue.head(10))

print("\n6. TIME PATTERNS")
print(f"   Payment types: {df_new['cash_type'].value_counts().to_dict()}")
print(f"\n   Time of day: {df_new['Time_of_Day'].value_counts().to_dict()}")
print(f"\n   Revenue by weekday:")
weekday_revenue = df_new.groupby('Weekday')['money'].sum().sort_values(ascending=False)
print(weekday_revenue)

# Check for trend
daily_revenue['day_num'] = range(len(daily_revenue))
correlation = daily_revenue[['day_num', 'revenue']].corr().iloc[0, 1]
growth = (daily_revenue['revenue'].iloc[-1] - daily_revenue['revenue'].iloc[0]) / daily_revenue['revenue'].iloc[0] * 100

print("\n7. TREND ANALYSIS")
print(f"   Correlation with time: {correlation:.4f}")
print(f"   Overall growth: {growth:.2f}%")
if abs(correlation) > 0.3:
    print(f"   ⚠️  STRONG TREND DETECTED (correlation={correlation:.2f})")
    print(f"   ⚠️  R² may be negative with this dataset too!")
else:
    print(f"   ✅ WEAK/NO TREND (correlation={correlation:.2f})")
    print(f"   ✅ R² should be positive!")

# Stationarity check
print("\n8. STATIONARITY TEST")
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(daily_revenue['revenue'].values)
print(f"   ADF Statistic: {adf_result[0]:.4f}")
print(f"   p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print(f"   ✅ STATIONARY (p < 0.05) - Good for R²!")
else:
    print(f"   ⚠️  NON-STATIONARY (p >= 0.05) - R² may be negative!")

print("\n" + "="*80)
print("COMPARISON WITH OLD DATASET")
print("="*80)

# Load old dataset summary
print("\nOLD DATASET (Coffee Shop Sales.xlsx):")
print("   - Total transactions: 149,116")
print("   - Total days: 181")
print("   - Total revenue: $698,812")
print("   - Mean daily revenue: $3,860")
print("   - Stores: 3")
print("   - Product categories: 9")
print("   - Growth: +124.4% (STRONG TREND)")
print("   - ADF p-value: 0.8445 (NON-STATIONARY)")
print("   - R²: -0.03 (negative)")

print(f"\nNEW DATASET (Coffe_sales.csv):")
print(f"   - Total transactions: {len(df_new):,}")
print(f"   - Total days: {len(daily_revenue)}")
print(f"   - Total revenue: ${df_new['money'].sum():,.2f}")
print(f"   - Mean daily revenue: ${daily_revenue['revenue'].mean():,.2f}")
print(f"   - Stores: 1")
print(f"   - Product categories: {df_new['coffee_name'].nunique()}")
print(f"   - Growth: {growth:.2f}%")
print(f"   - ADF p-value: {adf_result[1]:.4f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if abs(correlation) < 0.3 and adf_result[1] < 0.05:
    print("✅ NEW DATASET LOOKS PROMISING!")
    print("   - Stationary (good for R²)")
    print("   - Weak/no trend")
    print("   - Should give positive R²")
    print("\n   👉 RECOMMEND: Try building model with new dataset")
else:
    print("⚠️  NEW DATASET HAS SIMILAR ISSUES!")
    print(f"   - Correlation: {correlation:.2f}")
    print(f"   - ADF p-value: {adf_result[1]:.4f}")
    print("   - Likely will have negative R² too")
    print("\n   👉 RECOMMEND: Stick with old dataset (already 90% done)")

# Save daily revenue for modeling
daily_revenue.to_csv('/home/user/Coffee-shop/data/processed/new_daily_revenue.csv', index=False)
print(f"\n✅ Saved daily revenue to: data/processed/new_daily_revenue.csv")
