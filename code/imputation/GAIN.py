import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 啟用 IterativeImputer（此處保留以避免其他依賴）
from sklearn.impute import IterativeImputer  # 只保留GAIN時可不使用，但保留作為參考
from sklearn.decomposition import NMF  # 不再使用，可保留或移除
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ---------------------------
# 資料清洗與驗證函式
# ---------------------------
def clean_and_validate_data(df):
    """清洗資料並設定合理的值域限制，包括新增 iq 的合理範圍 (0 ~ 4500)"""
    df_clean = df.copy()
    mask_height = (df_clean['height'] < 140) | (df_clean['height'] > 200)
    df_clean.loc[mask_height, 'height'] = np.nan
    mask_weight = (df_clean['weight'] < 35) | (df_clean['weight'] > 120)
    df_clean.loc[mask_weight, 'weight'] = np.nan
    mask_phone = ~df_clean['phone_os'].isin([1.0, 2.0])
    df_clean.loc[mask_phone, 'phone_os'] = np.nan
    mask_fb = (df_clean['fb_friends'] < 0) | (df_clean['fb_friends'] > 3000)
    df_clean.loc[mask_fb, 'fb_friends'] = np.nan
    mask_yt = (df_clean['yt'] < 0) | (df_clean['yt'] > 24)
    df_clean.loc[mask_yt, 'yt'] = np.nan
    mask_star_sign = (df_clean['star_sign'] < 1) | (df_clean['star_sign'] > 12)
    df_clean.loc[mask_star_sign, 'star_sign'] = np.nan
    mask_sleepiness = (df_clean['sleepiness'] < 1) | (df_clean['sleepiness'] > 5)
    df_clean.loc[mask_sleepiness, 'sleepiness'] = np.nan
    # 新增：IQ 合理範圍限制
    if 'iq' in df_clean.columns:
        mask_iq = (df_clean['iq'] < 0) | (df_clean['iq'] > 4500)
        df_clean.loc[mask_iq, 'iq'] = np.nan
    return df_clean

def validate_imputed_values(df):
    """驗證並修正補值結果，包含 IQ 的合理區間"""
    df_valid = df.copy()
    df_valid['height'] = df_valid['height'].clip(140, 200)
    df_valid['weight'] = df_valid['weight'].clip(35, 120)
    df_valid['phone_os'] = np.round(df_valid['phone_os']).clip(1, 2)
    df_valid['fb_friends'] = df_valid['fb_friends'].clip(0, 3000)
    df_valid['yt'] = df_valid['yt'].clip(0, 24)
    df_valid['star_sign'] = np.round(df_valid['star_sign']).clip(1, 12)
    df_valid['sleepiness'] = np.round(df_valid['sleepiness']).clip(1, 5)
    if 'iq' in df_valid.columns:
        df_valid['iq'] = df_valid['iq'].clip(0, 4500)
    return df_valid

# ---------------------------
# outlier 調整函式：利用 IQR 方法將 outlier 設為缺失
# ---------------------------
def remove_outliers(df, cols):
    df_adjusted = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.15)
        Q3 = df[col].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_adjusted.loc[(df_adjusted[col] < lower_bound) | (df_adjusted[col] > upper_bound), col] = np.nan
    return df_adjusted

# ---------------------------
# GAIN 補值函式（使用777的GAIN模型實作）
# ---------------------------
def gain_imputation(df, batch_size=128, hint_rate=0.9, alpha=50, iterations=1500):
    """
    使用修改過架構的 GAIN 模型進行補值
    輸入 df 須為正規化後的資料 (數值介於 0~1)
    """
    X = df.values.astype(np.float32)
    M = 1 - np.isnan(X)  # 掩膜矩陣，缺失值為 0
    # 對缺失值進行隨機均勻填補
    X_filled = np.where(M==1, X, np.random.uniform(0, 1, X.shape).astype(np.float32))
    n_samples, n_features = X.shape

    # 修改後的生成器架構：增加層數、使用 LeakyReLU、BatchNormalization 及 Dropout
    input_dim = n_features * 2  # 輸入包含原始數據與 mask 串接
    generator = models.Sequential([
        layers.Dense(128, input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation='sigmoid')
    ])

    # 修改後的判別器架構
    discriminator = models.Sequential([
        layers.Dense(128, input_dim=n_features*2),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation='sigmoid')
    ])

    # 定義優化器
    gen_optimizer = optimizers.Adam(learning_rate=0.001)
    disc_optimizer = optimizers.Adam(learning_rate=0.001)

    # 訓練迴圈
    for it in range(iterations):
        batch_idx = np.random.choice(n_samples, batch_size, replace=True)
        X_batch = X_filled[batch_idx]
        M_batch = M[batch_idx]
        # Hint 機制：部分揭露 mask
        H_batch = np.random.binomial(1, hint_rate, size=(batch_size, n_features)).astype(np.float32)
        H_batch = M_batch * H_batch

        G_input = np.concatenate([X_batch, M_batch], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            G_sample = generator(G_input)
            # 僅針對缺失值取生成器的預測
            X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
            D_input = tf.concat([X_hat, H_batch], axis=1)
            D_prob = discriminator(D_input)
            # 判別器損失：真實值與生成值的辨識
            D_loss = -tf.reduce_mean(M_batch * tf.math.log(D_prob + 1e-8) + 
                                     (1 - M_batch) * tf.math.log(1 - D_prob + 1e-8))
            # 生成器損失：同時考慮騙過判別器與生成器輸出與真實值之間的差異
            G_loss = -tf.reduce_mean((1 - M_batch) * tf.math.log(D_prob + 1e-8)) + \
                     alpha * tf.reduce_mean(((M_batch * X_batch) - (M_batch * G_sample))**2)
        gradients_disc = tape.gradient(D_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
        gradients_gen = tape.gradient(G_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        if it % 100 == 0:
            print("Modified GAIN iteration {}: D_loss = {:.4f}, G_loss = {:.4f}".format(it, D_loss.numpy(), G_loss.numpy()))
        del tape

    # 使用訓練完的生成器進行補值
    G_input_full = np.concatenate([X_filled, M], axis=1)
    imputed_data = generator(G_input_full).numpy()
    imputed_result = M * X + (1 - M) * imputed_data
    df_imputed = pd.DataFrame(imputed_result, columns=df.columns, index=df.index)
    return df_imputed


# ---------------------------
# 主程式：依性別分組進行GAIN補值
# ---------------------------
def main():
    # 讀取資料（請依實際路徑調整 input_file）
    input_file = r"D:\datascience\Sec_homework\dataset\origin.csv"
    df = pd.read_csv(input_file, index_col=0)
    
    # 轉換 phone_os 欄位：apple->1, android->2
    df['phone_os'] = df['phone_os'].str.lower().str.strip().map({'apple': 1, 'android': 2})
    
    imputed_list = []  # 儲存各性別補值結果
    for gender_val in [1, 2]:
        print("開始處理 gender =", gender_val)
        # 過濾出該性別資料
        df_gender = df[df['gender'] == gender_val].copy()
        # 分離 self_intro 欄位（補值後再合併），並移除 gender 與 self_intro 以供補值
        df_self_intro = df_gender['self_intro']
        df_impute = df_gender.drop(['self_intro', 'gender'], axis=1)
        
        # 將異常值 '#NUM!' 替換為缺失，並轉為數值型態
        df_impute = df_impute.replace('#NUM!', np.nan)
        df_impute = df_impute.apply(pd.to_numeric, errors='coerce')
        # 清洗資料：含合理範圍檢查（包含 IQ）
        df_clean = clean_and_validate_data(df_impute)
        # 調整 outlier：不在 IQR 範圍內的數值視為缺失
        df_clean = remove_outliers(df_clean, df_clean.columns)
        
        # 為 GAIN 補值建立 scaler：以各欄位平均值填補後進行正規化
        fillna_values = df_clean.mean()
        scaler = MinMaxScaler()
        df_for_scaling = df_clean.fillna(fillna_values)
        scaler.fit(df_for_scaling)
        
        # 正規化資料
        df_norm = pd.DataFrame(scaler.transform(df_for_scaling), columns=df_clean.columns, index=df_clean.index)
        
        # 執行 GAIN 補值（補值函式中會印出訓練過程）
        df_gain_norm = gain_imputation(df_norm, iterations=1500)
        # 將補值結果轉回原始尺度
        df_gain = pd.DataFrame(scaler.inverse_transform(df_gain_norm), columns=df_norm.columns, index=df_norm.index)
        df_gain = validate_imputed_values(df_gain)
        
        # 合併回 self_intro 與 gender 欄位
        df_gain['self_intro'] = df_self_intro.values
        df_gain['gender'] = gender_val
        
        imputed_list.append(df_gain)
    
    # 合併兩個性別的結果
    df_final = pd.concat(imputed_list, axis=0)
    df_final.sort_index(inplace=True)
    
    # 輸出結果至 CSV（請依實際路徑調整 output_file）
    output_file = r"D:\datascience\Sec_homework\output_gain_gender_sep.csv"
    df_final.to_csv(output_file, index=False)
    print("GAIN補值結果已輸出至：", output_file)

if __name__ == "__main__":
    main()
