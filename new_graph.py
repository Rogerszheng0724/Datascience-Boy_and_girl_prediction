import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ========== Utility Functions (Keep original functions) ==========

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
    """
    Clean the data and set a proper range for each column.
    """
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
    # Check that star_sign is between 1 and 12
    mask_star_sign = (df_clean['star_sign'] < 1) | (df_clean['star_sign'] > 12)
    df_clean.loc[mask_star_sign, 'star_sign'] = np.nan
    # Check that sleepiness is between 1 and 5
    mask_sleepiness = (df_clean['sleepiness'] < 1) | (df_clean['sleepiness'] > 5)
    df_clean.loc[mask_sleepiness, 'sleepiness'] = np.nan
    return df_clean

def validate_imputed_values(df):
    """
    Validate and adjust the imputed results.
    """
    df_valid = df.copy()
    df_valid['height'] = df_valid['height'].clip(140, 200)
    df_valid['weight'] = df_valid['weight'].clip(35, 120)
    df_valid['phone_os'] = np.round(df_valid['phone_os']).clip(1, 2)
    df_valid['fb_friends'] = df_valid['fb_friends'].clip(0, 3000)
    df_valid['yt'] = df_valid['yt'].clip(0, 24)
    df_valid['star_sign'] = np.round(df_valid['star_sign']).clip(1, 12)
    df_valid['sleepiness'] = np.round(df_valid['sleepiness']).clip(1, 5)
    return df_valid

# ===========================
# Functions for adding mask and evaluation
# ===========================
def add_mask(df, mask_ratio=0.2, random_state=42):
    """
    For each column randomly select mask_ratio portion of indices to set as NaN.
    Returns the masked DataFrame and a dictionary with mask indices for each column.
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
    Compute the MSE and RMSE for each column (only for the artificially masked values).
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
# Deep Learning Imputation Method (GAIN) based on normalized data
# ===========================
def gain_imputation(df, batch_size=128, hint_rate=0.9, alpha=50, iterations=1500):
    """
    Modified GAIN imputation method (data must be normalized to [0,1]).
    Uses a network with additional layers, LeakyReLU, BatchNormalization, and Dropout.
    """
    X = df.values.astype(np.float32)
    M = 1 - np.isnan(X)  # mask matrix, missing values are 0
    # Fill missing values with random uniform values (data is normalized)
    X_filled = np.where(M == 1, X, np.random.uniform(0, 1, X.shape).astype(np.float32))
    n_samples, n_features = X.shape
    input_dim = n_features * 2  # input is concatenation of data and mask

    # Generator
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

    # Discriminator
    discriminator = models.Sequential([
        layers.Dense(128, input_dim=n_features * 2),
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

    gen_optimizer = optimizers.Adam(learning_rate=0.001)
    disc_optimizer = optimizers.Adam(learning_rate=0.001)

    # Training loop
    for it in range(iterations):
        batch_idx = np.random.choice(n_samples, batch_size, replace=True)
        X_batch = X_filled[batch_idx]
        M_batch = M[batch_idx]
        # Hint mechanism: randomly reveal part of the mask based on hint_rate
        H_batch = np.random.binomial(1, hint_rate, size=(batch_size, n_features)).astype(np.float32)
        H_batch = M_batch * H_batch
        # Concatenate original data with mask for generator input
        G_input = np.concatenate([X_batch, M_batch], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            G_sample = generator(G_input)
            # Generator only predicts missing values
            X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
            D_input = tf.concat([X_hat, H_batch], axis=1)
            D_prob = discriminator(D_input)
            # Discriminator loss: differentiate between real and generated data
            D_loss = -tf.reduce_mean(M_batch * tf.math.log(D_prob + 1e-8) + 
                                     (1 - M_batch) * tf.math.log(1 - D_prob + 1e-8))
            # Generator loss: try to fool discriminator and approximate true values (balanced by alpha)
            G_loss = -tf.reduce_mean((1 - M_batch) * tf.math.log(D_prob + 1e-8)) + \
                     alpha * tf.reduce_mean(((M_batch * X_batch) - (M_batch * G_sample))**2)
        gradients_disc = tape.gradient(D_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
        gradients_gen = tape.gradient(G_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        if it % 100 == 0:
            print("Modified GAIN iteration {}: D_loss = {:.4f}, G_loss = {:.4f}".format(it, D_loss.numpy(), G_loss.numpy()))
        del tape

    # Use the trained generator for imputation
    G_input_full = np.concatenate([X_filled, M], axis=1)
    imputed_data = generator(G_input_full).numpy()
    imputed_result = M * X + (1 - M) * imputed_data
    df_imputed = pd.DataFrame(imputed_result, columns=df.columns, index=df.index)
    return df_imputed

def autoencoder_imputation(df, epochs=200, batch_size=128):
    """
    Impute missing values using an autoencoder (data should be normalized).
    Train for 200 epochs and include Dropout layers.
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
    Impute missing values using an RNN autoencoder (data should be normalized).
    Train for 200 epochs and include dropout in the LSTM.
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

# ====================================================

def plot_imputation_comparison(eval_methods):
    """
    Plot a bar chart comparing the average RMSE for the given imputation methods.
    The methods are sorted in descending order (largest to smallest average RMSE).
    eval_methods: dictionary in the form {method_name: {column: {"MSE", "RMSE"}}}
    """
    avg_rmse = {}
    for method, metrics in eval_methods.items():
        # Calculate the average RMSE across all columns
        rmse_list = [score["RMSE"] for score in metrics.values()]
        avg_rmse[method] = np.mean(rmse_list)

    # Sort methods by average RMSE descending
    sorted_methods = sorted(avg_rmse.items(), key=lambda x: x[1], reverse=True)
    x_labels = [item[0] for item in sorted_methods]
    y_rmse = [item[1] for item in sorted_methods]

    # Plotting
    plt.figure(figsize=(6, 5), dpi=100)  # Adjust figure size as needed
    bars = plt.bar(x_labels, y_rmse, color='skyblue', edgecolor='black')
    plt.xlabel("Imputation Method")
    plt.ylabel("Average RMSE")
    plt.title("Gender 1 - Comparison of 5 Imputation Methods (Descending Order)")

    # Display the numerical value above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,  # leave some space between the bar and the text
            f"{height:.1f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    # Uncomment the following line to save the plot as a file:
    # plt.savefig("imputation_comparison.png", bbox_inches='tight')
    plt.show()

def main():
    # Change the path to your CSV file
    input_file = r"D:\datascience\Sec_homework\dataset\origin.csv"
    df = pd.read_csv(input_file, index_col=0)

    # Convert phone_os strings to numbers (apple->1, android->2)
    df['phone_os'] = df['phone_os'].str.lower().str.strip().map({'apple': 1, 'android': 2})

    # Only consider data where gender == 1 as an example
    gender_val = 1
    df_gender = df[df['gender'] == gender_val].copy()

    # Select the columns for imputation
    target_cols = ['height', 'weight', 'phone_os', 'fb_friends', 'yt', 'iq', 'star_sign', 'sleepiness']
    df_for_imputation = df_gender.drop(['self_intro', 'gender'], axis=1)[target_cols]
    df_for_imputation = df_for_imputation.replace('#NUM!', np.nan)
    df_for_imputation = df_for_imputation.apply(pd.to_numeric, errors='coerce')

    # Data cleaning and add artificial mask
    df_clean = clean_and_validate_data(df_for_imputation)
    df_masked, mask_info = add_mask(df_clean, mask_ratio=0.2, random_state=42)

    # ====== Keep Only 5 Methods ======
    # 1. Median Imputation
    df_median = df_masked.copy()
    for col in df_median.columns:
        median_val = df_clean[col].median()
        df_median[col] = df_median[col].fillna(median_val)
    df_median = validate_imputed_values(df_median)

    # 2. Mean Imputation
    df_mean = df_masked.copy()
    for col in df_mean.columns:
        mean_val = df_clean[col].mean()
        df_mean[col] = df_mean[col].fillna(mean_val)
    df_mean = validate_imputed_values(df_mean)

    # 3. KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_knn = knn_imputer.fit_transform(df_masked)
    df_knn = pd.DataFrame(imputed_knn, columns=df_masked.columns, index=df_masked.index)
    df_knn = validate_imputed_values(df_knn)

    # 4. EM Imputation
    df_em = em_imputation(df_masked)
    df_em = validate_imputed_values(df_em)

    # 5. GAIN Imputation (requires normalization)
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
    df_gain_norm = gain_imputation(df_masked_norm, iterations=500)  # Reduced iteration count for speed
    df_gain = pd.DataFrame(scaler.inverse_transform(df_gain_norm), columns=df_masked.columns, index=df_masked.index)
    df_gain = validate_imputed_values(df_gain)

    # Consolidate the 5 imputation methods
    imputation_methods = {
        'Median': df_median,
        'Mean': df_mean,
        'KNN': df_knn,
        'EM': df_em,
        'GAIN': df_gain
    }

    # Compute RMSE for each method
    eval_methods = {}
    for method_name, imputed_df in imputation_methods.items():
        eval_methods[method_name] = evaluate_imputation(df_clean, imputed_df, mask_info)

    # Plot the comparison chart
    plot_imputation_comparison(eval_methods)

if __name__ == "__main__":
    main()
