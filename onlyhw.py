import pandas as pd
import numpy as np
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer  # 使用 SentenceTransformer 取得語意嵌入
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# 載入 TensorFlow 用來建立 autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 若尚未下載 nltk 資源，可取消下列註解執行一次
nltk.download('stopwords')
nltk.download('punkt')

english_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. 小寫化
    text = text.lower()
    # 2. 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    # 3. 移除 URL
    text = re.sub(r'http\S+', '', text)
    # 4. 數字處理：將數字替換成 "NUM"
    text = re.sub(r'\d+', 'NUM', text)
    # 5. 移除標點符號
    text = re.sub(r'[^\w\s]', '', text)
    # 6. 清除多餘空白與控制字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. 拼寫檢查與修正（此步驟可能耗時，可根據需求選擇是否啟用）
    try:
        from textblob import TextBlob
        text = str(TextBlob(text).correct())
    except Exception as e:
        pass

    # 8. 縮寫展開
    abbrev_dict = {
        "u": "you",
        "r": "are",
        "btw": "by the way",
        "idk": "i do not know",
        "lol": "laugh out loud",
        "brb": "be right back",
        "im": "i am",
        "dont": "do not",
        "cant": "cannot"
    }
    tokens = word_tokenize(text)
    tokens = [abbrev_dict.get(token, token) for token in tokens]
    
    # 9. 移除停止詞
    tokens = [token for token in tokens if token.lower() not in english_stopwords]
    
    return " ".join(tokens)

# 讀取資料時只載入需要的欄位
train_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\train.csv', usecols=['height', 'weight', 'self_intro', 'gender'])
test_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\test.csv', usecols=['height', 'weight', 'self_intro'])

# 填補 self_intro 缺失值
train_df['self_intro'] = train_df['self_intro'].fillna("")
test_df['self_intro'] = test_df['self_intro'].fillna("")

print("進行文字前處理...")
train_df['self_intro'] = train_df['self_intro'].apply(preprocess_text)
test_df['self_intro'] = test_df['self_intro'].apply(preprocess_text)

# 僅保留數值特徵： height 與 weight
num_cols = ['height', 'weight']

# 轉換 gender 為二元標籤：假設 1 (男) 轉為 0，2 (女) 轉為 1
train_df['gender_bin'] = train_df['gender'].apply(lambda x: 0 if x == 1 else 1)

# 使用 SentenceTransformer 取得 self_intro 的語意嵌入
model_name = "all-distilroberta-v1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(model_name, device=device)
print("SentenceTransformer 使用裝置:", embedder.device)

print("計算 train 與 test 的 self_intro 嵌入...")
start_time = time.time()
train_embeddings = embedder.encode(
    train_df['self_intro'].tolist(),
    batch_size=64,
    show_progress_bar=True
)
test_embeddings = embedder.encode(
    test_df['self_intro'].tolist(),
    batch_size=64,
    show_progress_bar=True
)
print(f"嵌入完成，耗時: {time.time() - start_time:.2f} 秒")

# 建立 Autoencoder 來做降維，這裡假設原始嵌入維度是 embed_dim
embed_dim = train_embeddings.shape[1]
encoded_dim = 32  # 可根據需求調整

# 定義 Autoencoder 架構
input_layer = Input(shape=(embed_dim,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(256, activation='relu')(encoded)
bottleneck = Dense(encoded_dim, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(bottleneck)
decoded = Dense(512, activation='relu')(decoded)
output_layer = Dense(embed_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer='adam', loss='mse')
print("開始訓練 Autoencoder...")
# 加入 EarlyStopping，避免過度訓練
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

autoencoder.fit(
    train_embeddings, train_embeddings,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 利用訓練好的 encoder 轉換嵌入
train_embeddings_ae = encoder.predict(train_embeddings)
test_embeddings_ae = encoder.predict(test_embeddings)
print("Autoencoder 降維完成：原始維度 {} 降至 {} 維".format(embed_dim, encoded_dim))

# 將 Autoencoder 後的結果轉成 DataFrame
train_embed_df = pd.DataFrame(train_embeddings_ae, columns=[f"ae_embed_{i}" for i in range(encoded_dim)], index=train_df.index)
test_embed_df = pd.DataFrame(test_embeddings_ae, columns=[f"ae_embed_{i}" for i in range(encoded_dim)], index=test_df.index)

# 整合數值特徵與降維後的文字嵌入
train_features = pd.concat([train_df[num_cols].reset_index(drop=True),
                            train_embed_df.reset_index(drop=True)], axis=1)
test_features = pd.concat([test_df[num_cols].reset_index(drop=True),
                           test_embed_df.reset_index(drop=True)], axis=1)

# 切分訓練集與驗證集
X = train_features.copy()
y = train_df['gender_bin']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 使用 SMOTE 平衡資料
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 建立 Pipeline：Autoencoder 降維後仍然用 SelectKBest 選擇重要特徵
pipeline = Pipeline([
    ('select', SelectKBest(score_func=f_classif, k=150)),
    ('clf', XGBClassifier(eval_metric='logloss', random_state=42, tree_method='gpu_hist', n_jobs=-1))
])
# GridSearchCV 調參的參數空間
param_grid = {
    'clf__n_estimators': [100, 300],
    'clf__max_depth': [3, 7],
    'clf__learning_rate': [0.05, 0.1],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0],
    'clf__gamma': [0, 0.1]
}

print("\n進行 Pipeline 的 GridSearchCV 調參...")
start_model_time = time.time()
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=0, n_jobs=1)
grid.fit(X_train, y_train)
elapsed = time.time() - start_model_time

print("最佳參數:", grid.best_params_)
best_model = grid.best_estimator_
y_pred_val = best_model.predict(X_val)
acc = accuracy_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)
print("驗證集結果: Accuracy = {:.4f}, F1 = {:.4f}，耗時 {:.2f} 秒".format(acc, f1, elapsed))

# 使用最佳模型對 test 資料進行預測
test_pred = best_model.predict(test_features)
# 若原本的 gender 標籤為 1 與 2，此處加回 1
test_pred_converted = test_pred + 1

# 檢查是否有 id 欄位，若無則用 index 作為 id
if 'id' in test_df.columns:
    submission = pd.DataFrame({'id': test_df['id'], 'gender': test_pred_converted})
else:
    submission = pd.DataFrame({'id': test_df.index, 'gender': test_pred_converted})
submission.to_csv('submission.csv', index=False)
print("\n預測結果已儲存至 submission.csv")
