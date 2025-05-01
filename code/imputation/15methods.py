import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 啟用 IterativeImputer 功能
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ===========================
# 補值方法（原始函式）
# ===========================
def hot_deck_imputation(df):
    df_imputed = df.copy()
    for col in df_imputed.columns:
        missing_idx = df_imputed[col].isnull()
        non_missing_values = df_imputed.loc[~missing_idx, col]
        if non_missing_values.shape[0] > 0:
            df_imputed.loc[missing_idx, col] = np.random.choice(
                non_missing_values, size=missing_idx.sum(), replace=True)
    return df_imputed

def em_imputation(df, tol=1e-4, max_iter=100):
    df_em = df.copy()
    col_means = df_em.mean()
    df_em_filled = df_em.fillna(col_means)
    X = df_em_filled.values
    missing_mask = df_em.isnull().values
    prev_X = np.copy(X)
    for iteration in range(max_iter):
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X, rowvar=False)
        for i in range(X.shape[0]):
            if not np.any(missing_mask[i]):
                continue
            observed_indices = np.where(~missing_mask[i])[0]
            missing_indices = np.where(missing_mask[i])[0]
            if observed_indices.size == 0 or missing_indices.size == 0:
                continue
            mu_obs = mu[observed_indices]
            mu_miss = mu[missing_indices]
            Sigma_obs = Sigma[np.ix_(observed_indices, observed_indices)]
            if Sigma_obs.size == 0:
                continue
            try:
                cond_number = np.linalg.cond(Sigma_obs)
            except np.linalg.LinAlgError:
                cond_number = np.inf
            if cond_number > 1e12:
                Sigma_obs += np.eye(Sigma_obs.shape[0]) * 1e-6
            Sigma_miss_obs = Sigma[np.ix_(missing_indices, observed_indices)]
            x_obs = X[i, observed_indices]
            try:
                inv_Sigma_obs = np.linalg.inv(Sigma_obs)
            except np.linalg.LinAlgError:
                continue
            cond_expectation = mu_miss + Sigma_miss_obs.dot(inv_Sigma_obs).dot(x_obs - mu_obs)
            X[i, missing_indices] = cond_expectation
        if np.linalg.norm(X - prev_X) < tol:
            break
        prev_X = np.copy(X)
    df_imputed = pd.DataFrame(X, columns=df.columns, index=df.index)
    return df_imputed

def nmf_imputation(df, n_components=5, max_iter=1000):
    df_nmf = df.copy()
    df_temp = df_nmf.fillna(0)
    min_val = df_temp.min().min()
    shift = 0
    if min_val < 0:
        shift = abs(min_val)
        df_temp += shift
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=max_iter)
    W = model.fit_transform(df_temp)
    H = model.components_
    df_reconstructed = np.dot(W, H)
    if shift != 0:
        df_reconstructed -= shift
    df_imputed = df_nmf.copy()
    for i, col in enumerate(df_nmf.columns):
        mask = df_nmf[col].isnull()
        df_imputed.loc[mask, col] = df_reconstructed[:, i][mask]
    df_imputed.index = df_nmf.index
    return df_imputed

def clean_and_validate_data(df):
    """清洗資料並設定合理的值域限制"""
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
    return df_clean

def validate_imputed_values(df):
    """驗證並修正補值結果"""
    df_valid = df.copy()
    df_valid['height'] = df_valid['height'].clip(140, 200)
    df_valid['weight'] = df_valid['weight'].clip(35, 120)
    df_valid['phone_os'] = np.round(df_valid['phone_os']).clip(1, 2)
    df_valid['fb_friends'] = df_valid['fb_friends'].clip(0, 3000)
    df_valid['yt'] = df_valid['yt'].clip(0, 24)
    return df_valid

# ===========================
# 人工 mask 與評估函式
# ===========================
def add_mask(df, mask_ratio=0.2, random_state=42):
    """
    對每個欄位隨機選取 mask_ratio 比例的索引設為 NaN，
    並回傳 mask 後的 DataFrame 與各欄位被 mask 的索引資訊。
    """
    np.random.seed(random_state)
    df_masked = df.copy()
    mask_info = {}
    for col in df.columns:
        valid_idx = df.index[df[col].notnull()]
        n_mask = int(len(valid_idx) * mask_ratio)
        masked_indices = np.random.choice(valid_idx, size=n_mask, replace=False)
        mask_info[col] = masked_indices
        df_masked.loc[masked_indices, col] = np.nan
    return df_masked, mask_info

def evaluate_imputation(original_df, imputed_df, mask_info):
    """
    對每一個欄位（僅針對人工設置的缺失值）計算 MSE 與 RMSE
    """
    evaluation = {}
    for col, indices in mask_info.items():
        true_values = original_df.loc[indices, col]
        imputed_values = imputed_df.loc[indices, col]
        mse = mean_squared_error(true_values, imputed_values)
        rmse = np.sqrt(mse)
        evaluation[col] = {"MSE": mse, "RMSE": rmse}
    return evaluation

# ===========================
# 深度學習補值方法（基於正規化數據）
# ===========================

def gain_imputation(df, batch_size=128, hint_rate=0.9, alpha=50, iterations=1500):
    """
    簡化版 GAIN 補值方法（資料需為正規化後的數據）
    調整 alpha 為 50，並增加迭代次數至 1500
    """
    X = df.values.astype(np.float32)
    M = 1 - np.isnan(X)
    # GAIN 採用隨機均勻補值（資料已正規化在 [0,1]）
    X_filled = np.where(M==1, X, np.random.uniform(0, 1, X.shape).astype(np.float32))
    n_samples, n_features = X.shape
    input_dim = n_features * 2  # 資料與 mask 串接

    generator = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_features, activation='sigmoid')
    ])
    discriminator = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=n_features*2),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_features, activation='sigmoid')
    ])
    gen_optimizer = optimizers.Adam(learning_rate=0.001)
    disc_optimizer = optimizers.Adam(learning_rate=0.001)

    for it in range(iterations):
        batch_idx = np.random.choice(n_samples, batch_size, replace=True)
        X_batch = X_filled[batch_idx]
        M_batch = M[batch_idx]
        H_batch = np.random.binomial(1, hint_rate, size=(batch_size, n_features)).astype(np.float32)
        H_batch = M_batch * H_batch
        G_input = np.concatenate([X_batch, M_batch], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            G_sample = generator(G_input)
            X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
            D_input = tf.concat([X_hat, H_batch], axis=1)
            D_prob = discriminator(D_input)
            D_loss = -tf.reduce_mean(M_batch * tf.math.log(D_prob + 1e-8) + (1 - M_batch) * tf.math.log(1 - D_prob + 1e-8))
            G_loss = -tf.reduce_mean((1 - M_batch) * tf.math.log(D_prob + 1e-8)) + \
                     alpha * tf.reduce_mean(((M_batch * X_batch) - (M_batch * G_sample))**2)
        gradients_disc = tape.gradient(D_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
        gradients_gen = tape.gradient(G_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        if it % 100 == 0:
            print("GAIN iteration {}: D_loss = {:.4f}, G_loss = {:.4f}".format(it, D_loss.numpy(), G_loss.numpy()))
        del tape

    G_input_full = np.concatenate([X_filled, M], axis=1)
    imputed_data = generator(G_input_full).numpy()
    imputed_result = M * X + (1 - M) * imputed_data
    df_imputed = pd.DataFrame(imputed_result, columns=df.columns, index=df.index)
    return df_imputed

def autoencoder_imputation(df, epochs=200, batch_size=128):
    """
    利用自動編碼器進行補值（資料需為正規化後的數據）
    調整訓練 epochs 為 200，並加入 Dropout 層
    """
    X = df.values.astype(np.float32)
    M = 1 - np.isnan(X)
    col_mean = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_mean, X)
    n_samples, n_features = X.shape

    input_layer = layers.Input(shape=(n_features,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(n_features, activation='linear')(decoded)
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_filled, X_filled, epochs=epochs, batch_size=batch_size, verbose=0)
    X_reconstructed = autoencoder.predict(X_filled)
    imputed_result = M * X_filled + (1 - M) * X_reconstructed
    df_imputed = pd.DataFrame(imputed_result, columns=df.columns, index=df.index)
    return df_imputed

def rnn_imputation(df, epochs=200, batch_size=128):
    """
    利用 RNN 自動編碼器進行補值（資料需為正規化後的數據）
    調整訓練 epochs 為 200，並在 LSTM 中加入 dropout
    """
    X = df.values.astype(np.float32)
    M = 1 - np.isnan(X)
    col_mean = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_mean, X)
    n_samples, n_features = X.shape

    X_rnn = X_filled.reshape(n_samples, n_features, 1)
    inputs = layers.Input(shape=(n_features, 1))
    encoded = layers.LSTM(64, activation='relu', dropout=0.2, return_sequences=False)(inputs)
    decoded = layers.RepeatVector(n_features)(encoded)
    decoded = layers.LSTM(64, activation='relu', dropout=0.2, return_sequences=True)(decoded)
    outputs = layers.TimeDistributed(layers.Dense(1))(decoded)
    rnn_autoencoder = models.Model(inputs, outputs)
    rnn_autoencoder.compile(optimizer='adam', loss='mse')

    rnn_autoencoder.fit(X_rnn, X_rnn, epochs=epochs, batch_size=batch_size, verbose=0)
    X_reconstructed = rnn_autoencoder.predict(X_rnn)
    X_reconstructed = X_reconstructed.reshape(n_samples, n_features)
    imputed_result = M * X_filled + (1 - M) * X_reconstructed
    df_imputed = pd.DataFrame(imputed_result, columns=df.columns, index=df.index)
    return df_imputed

# ===========================
# 主程式
# ===========================
def main():
    # 讀取資料
    input_file = r"D:\datascience\Sec_homework\dataset\origin.csv"
    df = pd.read_csv(input_file, index_col=0)
    
    # 前處理：phone_os 轉換
    df['phone_os'] = df['phone_os'].str.lower().str.strip().map({'apple': 1, 'android': 2})
    
    # 分離 self_intro 並刪除 gender
    df_self_intro = df['self_intro']
    df_for_imputation = df.drop(['self_intro', 'gender'], axis=1)
    target_cols = ['height', 'weight', 'phone_os', 'fb_friends', 'yt', 'iq']
    df_for_imputation = df_for_imputation[target_cols]
    
    # 資料清洗
    df_for_imputation = df_for_imputation.replace('#NUM!', np.nan)
    df_for_imputation = df_for_imputation.apply(pd.to_numeric, errors='coerce')
    df_clean = clean_and_validate_data(df_for_imputation)
    
    # 加入 20% mask
    df_masked, mask_info = add_mask(df_clean, mask_ratio=0.2, random_state=42)
    
    # 傳統方法（原始尺度）
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_knn = knn_imputer.fit_transform(df_masked)
    df_knn = pd.DataFrame(imputed_knn, columns=df_masked.columns, index=df_masked.index)
    df_knn = validate_imputed_values(df_knn)
    
    mice_imputer = IterativeImputer(random_state=0)
    imputed_mice = mice_imputer.fit_transform(df_masked)
    df_mice = pd.DataFrame(imputed_mice, columns=df_masked.columns, index=df_masked.index)
    df_mice = validate_imputed_values(df_mice)
    
    df_em = em_imputation(df_masked)
    df_em = validate_imputed_values(df_em)
    
    df_hotdeck = hot_deck_imputation(df_masked)
    df_hotdeck = validate_imputed_values(df_hotdeck)
    
    df_nmf = nmf_imputation(df_masked)
    df_nmf = validate_imputed_values(df_nmf)
    
    # 深度學習方法：先正規化（以 df_clean 補缺後建立 scaler）
    fillna_values = df_clean.mean()
    scaler = MinMaxScaler()
    df_for_scaling = df_clean.fillna(fillna_values)
    scaler.fit(df_for_scaling)
    
    def normalize_df(df, fillna_values, scaler):
        df_filled = df.copy()
        for col in df.columns:
            df_filled[col] = df_filled[col].fillna(fillna_values[col])
        df_norm = pd.DataFrame(scaler.transform(df_filled), columns=df.columns, index=df.index)
        return df_norm

    df_masked_norm = normalize_df(df_masked, fillna_values, scaler)
    
    df_gain_norm = gain_imputation(df_masked_norm, iterations=1500)
    df_gain = pd.DataFrame(scaler.inverse_transform(df_gain_norm), columns=df_masked.columns, index=df_masked.index)
    df_gain = validate_imputed_values(df_gain)
    
    df_autoencoder_norm = autoencoder_imputation(df_masked_norm, epochs=200)
    df_autoencoder = pd.DataFrame(scaler.inverse_transform(df_autoencoder_norm), columns=df_masked.columns, index=df_masked.index)
    df_autoencoder = validate_imputed_values(df_autoencoder)
    
    df_rnn_norm = rnn_imputation(df_masked_norm, epochs=200)
    df_rnn = pd.DataFrame(scaler.inverse_transform(df_rnn_norm), columns=df_masked.columns, index=df_masked.index)
    df_rnn = validate_imputed_values(df_rnn)
    
    # ===== 定義 15 種方法 =====
    hybrid_methods = {}
    # 8 基本方法
    hybrid_methods['KNN'] = df_knn
    hybrid_methods['MICE'] = df_mice
    hybrid_methods['EM'] = df_em
    hybrid_methods['HotDeck'] = df_hotdeck
    hybrid_methods['NMF'] = df_nmf
    hybrid_methods['GAIN'] = df_gain
    hybrid_methods['Autoencoder'] = df_autoencoder
    hybrid_methods['RNN'] = df_rnn
    # 方法 9：深度學習平均
    hybrid_methods['DL_Avg'] = (df_gain + df_autoencoder + df_rnn) / 3
    # 方法 10：深度學習加權 1
    hybrid_methods['DL_Weighted_1'] = 0.4 * df_gain + 0.3 * df_autoencoder + 0.3 * df_rnn
    # 方法 11：深度學習加權 2
    hybrid_methods['DL_Weighted_2'] = 0.3 * df_gain + 0.4 * df_autoencoder + 0.3 * df_rnn
    # 方法 12：深度學習加權 3
    hybrid_methods['DL_Weighted_3'] = 0.3 * df_gain + 0.3 * df_autoencoder + 0.4 * df_rnn
    # 方法 13：傳統與深度學習混合 (平均傳統方法與 DL_Avg)
    trad_avg = (df_knn + df_mice + df_em + df_hotdeck + df_nmf) / 5
    hybrid_methods['Hybrid_Trad_DL'] = (trad_avg + hybrid_methods['DL_Avg']) / 2
    # 方法 14：深度學習中位數
    median_array = np.median(np.array([df_gain.values, df_autoencoder.values, df_rnn.values]), axis=0)
    hybrid_methods['DL_Median'] = pd.DataFrame(median_array, columns=df_masked.columns, index=df_masked.index)
    # 方法 15：所有方法均值融合
    hybrid_methods['Ensemble_All'] = (df_knn + df_mice + df_em + df_hotdeck + df_nmf + df_gain + df_autoencoder + df_rnn) / 8
    
    # ===== 各方法評估 =====
    # 建立字典儲存每種方法的評估結果
    eval_methods = {}
    for method_name, imputed_df in hybrid_methods.items():
        eval_methods[method_name] = evaluate_imputation(df_clean, imputed_df, mask_info)
    
    # ===== 對每個欄位選出最佳方法（以 RMSE 最低為準） =====
    summary_rows = []
    for col in df_clean.columns:
        best_method = None
        best_rmse = np.inf
        best_mse = None
        for method_name, metrics in eval_methods.items():
            if metrics[col]['RMSE'] < best_rmse:
                best_rmse = metrics[col]['RMSE']
                best_mse = metrics[col]['MSE']
                best_method = method_name
        summary_rows.append({
            '欄位': col,
            '最佳方法': best_method,
            'RMSE': best_rmse,
            'MSE': best_mse
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n最終最佳方法評估表：")
    print(summary_df)
    
if __name__ == "__main__":
    main()
