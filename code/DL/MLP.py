import os
import pandas as pd
import numpy as np
import re
import random
import time
from tqdm import tqdm

# ------------------ 載入翻譯與向量化相關套件 ------------------
from googletrans import Translator
from gensim.models import KeyedVectors
import gensim.downloader as api

# 載入其他機器學習相關套件
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------ 自訂函數區 ------------------
def contains_chinese(text):
    """檢查文字是否包含中文"""
    return bool(re.search('[\u4e00-\u9fff]', text))

# 建立翻譯器物件
translator = Translator()

def translate_text(text):
    """
    使用 googletrans 將中文翻譯成英文 (同步)
    """
    try:
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"翻譯失敗，錯誤：{e}")
        return text  # 若翻譯失敗，返回原始文字

def load_w2v_model():
    """
    載入 word2vec 模型，若無法載入則使用備援模型。
    """
    model_path = 'GoogleNews-vectors-negative300.bin'
    if os.path.exists(model_path):
        return KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        try:
            return api.load("glove-wiki-gigaword-300")
        except Exception as e:
            print("無法載入 glove-wiki-gigaword-300 模型，錯誤：", e)
            return api.load("glove-wiki-gigaword-50")

# 載入 word2vec 模型
w2v_model = load_w2v_model()

def vectorize_text(text):
    """
    使用 gensim 的預訓練模型將輸入文字轉換為向量。
    將文字拆解為單詞，取每個單詞的向量平均值。
    若無單詞在模型中，則回傳零向量。
    """
    words = text.split()
    valid_vectors = []
    for word in words:
        if word in w2v_model.key_to_index:
            valid_vectors.append(w2v_model[word])
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

def process_self_intro(text):
    """處理 self_intro 欄位：翻譯、向量化，若缺失則視為 'none'"""
    if pd.isnull(text):
        text = "none"
    if contains_chinese(text):
        text = translate_text(text)
    vec = vectorize_text(text)
    return vec

def expand_self_intro(df):
    """
    處理 DataFrame 中的 self_intro 欄位，
    並將其展開為向量維度的多個新特徵（例如 self_intro_0 ~ self_intro_n）。
    """
    vectors = df['self_intro'].apply(process_self_intro)
    vector_df = pd.DataFrame(vectors.tolist(), index=df.index)
    vector_df.columns = [f'self_intro_{i}' for i in range(vector_df.shape[1])]
    df = df.drop('self_intro', axis=1)
    df = pd.concat([df, vector_df], axis=1)
    return df

# ------------------ 讀取資料 ------------------
train = pd.read_csv(r'D:\datascience\Sec_homework\dataset\train.csv')
test = pd.read_csv(r'D:\datascience\Sec_homework\dataset\test.csv')

# ------------------ 1. 處理缺失值：只對訓練集填補 ------------------
cols_impute = ['height', 'weight', 'fb_friends', 'yt']
for dataset in [train, test]:
    for col in cols_impute:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

for col in cols_impute:
    median_gender1 = train.loc[train['gender'] == 1, col].median()
    median_gender2 = train.loc[train['gender'] == 2, col].median()
    train.loc[(train['gender'] == 1) & (train[col].isnull()), col] = median_gender1
    train.loc[(train['gender'] == 2) & (train[col].isnull()), col] = median_gender2

# ------------------ 2. 處理「星座」欄位 ------------------
zodiacs = ['牡羊座', '金牛座', '雙子座', '巨蟹座', '獅子座', '處女座',
           '天秤座', '天蠍座', '射手座', '摩羯座', '水瓶座', '雙魚座']
for dataset in [train, test]:
    dataset['star_sign'] = dataset['star_sign'].fillna(pd.Series(np.random.choice(zodiacs, size=len(dataset)), index=dataset.index))

# ------------------ 3. 處理「phone_os」欄位 ------------------
mapping = {'android': 0, 'iphone': 1}
for dataset in [train, test]:
    dataset['phone_os'] = dataset['phone_os'].map(mapping)
    missing_count = dataset['phone_os'].isnull().sum()
    if missing_count > 0:
        dataset.loc[dataset['phone_os'].isnull(), 'phone_os'] = np.random.choice([0, 1], size=missing_count)

# ------------------ 4. 處理「sleepness」欄位 ------------------
for dataset in [train, test]:
    missing_count = dataset['sleepiness'].isnull().sum()
    if missing_count > 0:
        dataset.loc[dataset['sleepiness'].isnull(), 'sleepiness'] = np.random.randint(1, 6, size=missing_count)

# ------------------ 5. 處理「IQ」欄位 ------------------
train['iq'] = train['iq'].fillna(0)
test['iq'] = test['iq'].fillna(0)

# ------------------ 6. 處理「self_intro」欄位：翻譯與向量化 ------------------
train = expand_self_intro(train)
test = expand_self_intro(test)

# ------------------ 7. One-hot Encoding & 特徵對齊 ------------------
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)
X_train_full = train.drop('gender', axis=1)
y = train['gender'] - 1  # 將原本的 1,2 轉換成 0,1 用於訓練
X_test = test.copy()
X_train_full, X_test = X_train_full.align(X_test, join='inner', axis=1)

# ------------------ 8. 資料清洗與極端值處理 ------------------
X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
X_train_full = X_train_full.fillna(X_train_full.median())
X_train_full = X_train_full.clip(-1e6, 1e6)

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(X_train_full.median())
X_test = X_test.clip(-1e6, 1e6)

# ------------------ 9. MLP 模型訓練與自動調參 ------------------
# 切分訓練資料成訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42)

# 引入 TensorFlow 及 Keras 套件
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tqdm.keras import TqdmCallback

# 全域變數：輸入特徵數量
input_dim = X_train.shape[1]

def create_model(hidden_layer_sizes=(64,), activation='relu', dropout_rate=0.0, optimizer='adam', learning_rate_init=0.001):
    """建立一個簡單的 MLP 模型"""
    model = Sequential()
    # 第一層
    model.add(Dense(hidden_layer_sizes[0], input_dim=input_dim, activation=activation))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    # 其他隱藏層
    for units in hidden_layer_sizes[1:]:
        model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    # 輸出層
    model.add(Dense(1, activation='sigmoid'))
    # 選擇最佳優化器
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate_init)
    else:
        opt = SGD(learning_rate=learning_rate_init)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 使用 KerasClassifier 包裝模型以便與 GridSearchCV 搭配
mlp_clf = KerasClassifier(build_fn=create_model, verbose=0)

# 定義參數網格
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32)],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.0, 0.2],
    'optimizer': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}

print("開始 GridSearchCV 調參 MLP...")
grid = GridSearchCV(estimator=mlp_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=0)
start = time.time()
# 利用 TqdmCallback 顯示每個 epoch 的進度
grid.fit(X_train, y_train, callbacks=[TqdmCallback(verbose=1)])
elapsed = time.time() - start
print(f"\nBest MLP parameters: {grid.best_params_} (Elapsed time: {elapsed:.2f} sec)")

best_mlp = grid.best_estimator_

# 評估驗證集
y_pred_val = best_mlp.predict(X_val)
print("\nMLP Classifier Accuracy on validation: {:.4f}".format(accuracy_score(y_val, y_pred_val)))
print("MLP Classifier Report:\n", classification_report(y_val, y_pred_val))

# 預測測試集 (將 0,1 轉回原本 1,2 格式)
test_pred = best_mlp.predict(X_test)
test_pred = test_pred + 1
if 'id' in test.columns:
    submission = pd.DataFrame({'id': test['id'], 'gender': test_pred})
else:
    submission = pd.DataFrame({'id': test.index, 'gender': test_pred})
submission.to_csv('submission.csv', index=False)
print("預測結果已儲存至 submission.csv")
