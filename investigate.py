import pandas as pd
import numpy as np

print("Loading data...")
df_old = pd.read_parquet("./nq_1min.parquet")
df_new = pd.read_parquet("./NQ_1min_output/continuous/NQ_continuous_raw.parquet").set_index("datetime")

# Strip timezones
df_old.index = df_old.index.tz_localize(None)
df_new.index = df_new.index.tz_localize(None)

# ==========================================================
# 1. AUTO-ALIGNMENT (Guarantees we compare the exact same trades)
# ==========================================================
best_shift = 0
max_matches = 0
for shift in [0, -1, 1]:
    temp_new = df_new.copy()
    temp_new.index = temp_new.index + pd.Timedelta(minutes=shift)
    common = df_old.index.intersection(temp_new.index)
    matches = (df_old.loc[common, 'close'] == temp_new.loc[common, 'close']).sum()
    if matches > max_matches:
        max_matches = matches
        best_shift = shift

df_new.index = df_new.index + pd.Timedelta(minutes=best_shift)
common_dates = df_old.index.intersection(df_new.index)
print(f"\nAuto-aligned NinjaTrader data by {best_shift} minutes.")

# ==========================================================
# 2. MISMATCH ANALYSIS (How big are the price differences?)
# ==========================================================
mismatches = common_dates[df_old.loc[common_dates, 'close'] != df_new.loc[common_dates, 'close']]
diffs = (df_old.loc[mismatches, 'close'] - df_new.loc[mismatches, 'close']).abs()

print("\n==================================================")
print(f"PRICE MISMATCH STATS (Total: {len(mismatches):,})")
print("==================================================")
if len(mismatches) > 0:
    print(f"Average Price Gap:       {diffs.mean():.2f} points")
    print(f"Median Price Gap:        {diffs.median():.2f} points")
    print(f"90% of gaps are under:   {diffs.quantile(0.90):.2f} points")
    print(f"99% of gaps are under:   {diffs.quantile(0.99):.2f} points")
    print(f"Maximum Price Gap:       {diffs.max():.2f} points")
    
    # Check if they are clustered in Rollover Months (Mar, Jun, Sep, Dec)
    rollover_months = mismatches.month.isin([3, 6, 9, 12])
    print(f"\nMismatches occurring during Rollover Months: {rollover_months.mean():.1%}")
else:
    print("0 Mismatches found.")

# ==========================================================
# 3. MISSING BARS ANALYSIS (Are we missing important volume?)
# ==========================================================
missing_idx = df_old.index.difference(df_new.index)
missing_vols = df_old.loc[missing_idx, 'volume']

print("\n==================================================")
print(f"MISSING BARS STATS (Total: {len(missing_idx):,})")
print("==================================================")
if len(missing_idx) > 0:
    print(f"Average Volume of Missing Bars:       {missing_vols.mean():.2f}")
    print(f"Median Volume of Missing Bars:        {missing_vols.median():.2f}")
    print(f"90% of missing bars have volume under: {missing_vols.quantile(0.90):.2f}")
    print(f"95% of missing bars have volume under: {missing_vols.quantile(0.95):.2f}")
    print(f"99% of missing bars have volume under: {missing_vols.quantile(0.99):.2f}")
    print(f"Maximum Volume in a single missing bar: {missing_vols.max():.2f}")
    
    # Calculate total impact
    total_old_vol = df_old['volume'].sum()
    total_missing_vol = missing_vols.sum()
    print(f"\nTotal Volume across all {len(df_old):,} old bars: {total_old_vol:,.0f}")
    print(f"Total Volume inside the missing bars:       {total_missing_vol:,.0f}")
    print(f"Percentage of total market volume missing:  {total_missing_vol / total_old_vol:.4%}")
else:
    print("0 Missing bars found.")