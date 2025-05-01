import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency

# 讀取資料 (請確認路徑正確)
df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\output_gain.csv')

# 1. 前處理
# 1.1 將 gender 轉換為二元變數：1 (男) -> 0, 2 (女) -> 1
df['gender_bin'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

# 1.2 將 self_intro 轉換為文字長度
df['self_intro_length'] = df['self_intro'].astype(str).apply(len)

# 2. 定義要計算的欄位及其型態
# 將 sleepiness 當作數值處理（雖然是有序類別）
cols = ['gender_bin', 'star_sign', 'phone_os', 'height', 'weight', 
        'sleepiness', 'iq', 'fb_friends', 'yt', 'self_intro_length']

# 定義每個欄位的型態
# 說明：'numeric' 表示連續數值；'binary' 表示二元數值；'categorical' 表示純類別型
col_types = {
    'gender_bin': 'binary',       # 已轉為 0/1
    'star_sign': 'categorical',   # 星座 (1~12) 但順序無意義
    'phone_os': 'categorical',    # Apple 或 Android
    'height': 'numeric',
    'weight': 'numeric',
    'sleepiness': 'numeric',      # 評分 1-5
    'iq': 'numeric',
    'fb_friends': 'numeric',
    'yt': 'numeric',
    'self_intro_length': 'numeric'
}

# 3. 定義各種相關性計算方法

# 3.1 Pearson 相關 (數值 vs 數值)
def pearson_corr(x, y):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return np.nan
    corr, _ = pearsonr(valid.iloc[:,0], valid.iloc[:,1])
    return corr

# 3.2 Correlation Ratio (數值 vs. 類別)
def correlation_ratio(categories, measurements):
    # 將類別變數因子化
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg = np.mean(measurements)
    numerator = 0
    denominator = np.sum((measurements - y_avg) ** 2)
    for i in range(cat_num):
        cat_measures = measurements[fcat == i]
        if len(cat_measures) == 0:
            continue
        numerator += len(cat_measures) * (np.mean(cat_measures) - y_avg) ** 2
    if denominator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)

# 3.3 Cramér’s V (類別 vs. 類別)
def cramers_v(x, y):
    valid = pd.concat([x, y], axis=1).dropna()
    table = pd.crosstab(valid.iloc[:,0], valid.iloc[:,1])
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# 3.4 根據欄位型態決定使用哪個方法計算兩兩相關性
def compute_association(col1, col2, type1, type2):
    # 取出非缺值資料
    x = df[col1]
    y = df[col2]
    valid = pd.concat([x, y], axis=1).dropna()
    if valid.empty:
        return np.nan

    # 兩個皆為數值型 (包含 binary)
    if type1 in ['numeric', 'binary'] and type2 in ['numeric', 'binary']:
        return pearson_corr(valid.iloc[:,0], valid.iloc[:,1])
    
    # 一個數值、一個類別 (以數值變數作為因變數)
    elif (type1 in ['numeric', 'binary'] and type2 == 'categorical'):
        return correlation_ratio(valid.iloc[:,1], valid.iloc[:,0])
    elif (type1 == 'categorical' and type2 in ['numeric', 'binary']):
        return correlation_ratio(valid.iloc[:,0], valid.iloc[:,1])
    
    # 兩個皆為類別
    elif type1 == 'categorical' and type2 == 'categorical':
        return cramers_v(valid.iloc[:,0], valid.iloc[:,1])
    
    else:
        return np.nan

# 4. 建立兩兩相關性矩陣
assoc_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

for col1 in cols:
    for col2 in cols:
        # 若相同欄位，設為 1
        if col1 == col2:
            assoc_matrix.loc[col1, col2] = 1.0
        else:
            assoc = compute_association(col1, col2, col_types[col1], col_types[col2])
            assoc_matrix.loc[col1, col2] = assoc

print("各欄位間的相關性/相似性矩陣：")
print(assoc_matrix)
