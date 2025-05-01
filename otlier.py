import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv(r"D:\datascience\Sec_homework\dataset\origin.csv")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

def remove_outliers_custom(df, cols, q_low, q_high):
    """
    Remove outliers for given columns based on specified quantiles.
    For each column:
      - Compute Q1 = quantile(q_low) and Q3 = quantile(q_high)
      - Calculate IQR = Q3 - Q1
      - Determine lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR
      - Replace values outside [lower_bound, upper_bound] with NaN
    Returns the adjusted dataframe and a dictionary of bounds.
    """
    df_adjusted = df.copy()
    bounds = {}
    for col in cols:
        Q1 = df[col].quantile(q_low)
        Q3 = df[col].quantile(q_high)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        bounds[col] = (lower_bound, upper_bound)
        df_adjusted.loc[(df_adjusted[col] < lower_bound) | (df_adjusted[col] > upper_bound), col] = np.nan
    return df_adjusted, bounds

# Define three scenarios with different quantile settings
scenarios = {
    "Scenario_1_0.2_0.8": (0.2, 0.8),
    "Scenario_2_0.15_0.85": (0.15, 0.85),
    "Scenario_3_0.25_0.75": (0.25, 0.75)
}

# List to collect results for CSV output
results = []

for scenario_name, (q_low, q_high) in scenarios.items():
    df_filtered, bounds = remove_outliers_custom(df, numeric_cols, q_low, q_high)
    for col in numeric_cols:
        lower_bound, upper_bound = bounds[col]
        normal_count = df_filtered[col].count()  # count of valid data points after outlier removal
        total_count = df[col].count()             # original count of valid data points
        results.append({
            "Column": col,
            "Scenario": scenario_name,
            "Quantile_Low": q_low,
            "Quantile_High": q_high,
            "LowerBound": lower_bound,
            "UpperBound": upper_bound,
            "NormalCount": normal_count,
            "TotalCount": total_count
        })

# Convert results to DataFrame and output to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("outlier_comparison.csv", index=False)

print("CSV file 'outlier_comparison.csv' has been generated with outlier comparison results.")
