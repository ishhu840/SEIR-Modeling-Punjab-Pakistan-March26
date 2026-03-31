import pandas as pd

optimal = pd.read_excel('../03_Results/V4_Optimal_Lags_ByCity.xlsx')
stats = pd.read_excel('../03_Results/V4_StatTests.xlsx')

print("=== KEY NUMBERS FOR CHAPTER 3 TEXT ===")
print(f"Mean Testing Correlation: {optimal['Test_r'].mean():.3f}")
print(f"Mean Reporting Fraction: {optimal['rho'].mean()*100:.1f}%")
print(f"Best City: {optimal.loc[optimal['Test_r'].idxmax(), 'City']} (r={optimal['Test_r'].max():.3f})")
print(f"Rain Lag Range: {optimal['Rain_Lag'].min()}-{optimal['Rain_Lag'].max()} weeks")
print(f"Temp Lag Range: {optimal['Temp_Lag'].min()}-{optimal['Temp_Lag'].max()} weeks")

print("\n=== PER-CITY TABLE DATA ===")
for _, row in stats.iterrows():
    ci = f"[{row['Test_CI_lower']:.3f}, {row['Test_CI_upper']:.3f}]"
    mae = f"{row['Test_MAE']:.1f}" if 'Test_MAE' in row else "N/A"
    rmse = f"{row['Test_RMSE']:.1f}" if 'Test_RMSE' in row else "N/A"
    print(f"{row['City']:20s} | Rain={row['Optimal_Rain_Lag']:2d} | Temp={row['Optimal_Temp_Lag']:2d} | ρ={row['rho_pct']:5.1f}% | r={row['Test_r']:.3f} {ci} | MAE={mae} | RMSE={rmse}")
