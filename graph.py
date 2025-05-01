import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, chi2_contingency

# ------------------ Load Data ------------------
df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\no_sep.csv')

# Convert gender to binary variable (0: male, 1: female)
df['gender_bin'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

# ------------------ Correlation Analysis for Continuous and Categorical Variables ------------------
# Continuous variables: Using Point-Biserial Correlation for numerical data
continuous_vars = ['height', 'weight', 'iq', 'fb_friends', 'yt']
pb_results = {}
for var in continuous_vars:
    valid = df[['gender_bin', var]].dropna()
    corr, p_value = pointbiserialr(valid[var], valid['gender_bin'])
    pb_results[var] = (corr, p_value)

print("Correlation of continuous variables (Point-Biserial) with gender:")
for var, (corr, p_value) in pb_results.items():
    print(f"  {var}: correlation = {corr:.3f}, p-value = {p_value:.3g}")

# Categorical variables: Calculate Cramér's V using Chi-square Test
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

categorical_vars = ['star_sign', 'phone_os', 'sleepiness']
cramers_results = {}
for var in categorical_vars:
    contingency = pd.crosstab(df['gender'], df[var])
    v = cramers_v(contingency)
    cramers_results[var] = v

print("\nCorrelation of categorical variables (Cramér's V) with gender:")
for var, v in cramers_results.items():
    print(f"  {var}: Cramér's V = {v:.3f}")

# ------------------ Heatmap Plotting ------------------
# Focus on continuous variables and gender_bin
cols = continuous_vars + ['gender_bin']
corr_matrix = df[cols].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
