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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 載入 TabNet 模型
from pytorch_tabnet.tab_model import TabNetClassifier

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
# 為避免 NaN 造成預測錯誤，以訓練集中位數填補測試集缺失值
X_test = X_test.fillna(X_train_full.median())
X_test = X_test.clip(-1e6, 1e6)

# ------------------ 9. 模型訓練：使用 TabNet 並手動 Grid Search 自動調參 ------------------
# 切分訓練資料成訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42)

# 強制轉換資料型態：TabNet 預期輸入為 float32，而目標值使用 int64
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)

# 設定 TabNet 的超參數網格
param_grid = {
    'n_d': [8, 16],
    'n_a': [8, 16],
    'n_steps': [3, 5],
    'gamma': [1.0, 1.5],
    'lambda_sparse': [1e-3, 1e-4],
    'learning_rate': [0.01, 0.001],
}

from itertools import product
keys = list(param_grid.keys())
combinations = list(product(*(param_grid[key] for key in keys)))

best_score = -np.inf
best_params = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("開始手動 Grid Search 調參 TabNetClassifier...")
grid_start = time.time()

# 透過 tqdm 顯示所有參數組合的進度
for params in tqdm(combinations, desc="GridSearch", total=len(combinations)):
    # 建立參數字典
    param_dict = dict(zip(keys, params))
    # 將 learning_rate 從主參數中取出，放入 optimizer_params
    lr = param_dict.pop('learning_rate')
    optimizer_params = dict(lr=lr)
    
    # 建立 TabNetClassifier (關閉內部 verbose，以免影響 tqdm 顯示)
    clf = TabNetClassifier(
         verbose=0,
         optimizer_params=optimizer_params,
         **param_dict
    )
    # 使用 cross_val_score 進行 5 折交叉驗證
    scores = cross_val_score(clf, X_train.values, y_train.values, cv=cv, scoring='accuracy', n_jobs=1)
    mean_score = np.mean(scores)
    if mean_score > best_score:
         best_score = mean_score
         best_params = dict(zip(keys, params))
    # 將 learning_rate 加回 best_params（若此組合較佳）
    if best_params is not None:
         best_params['learning_rate'] = best_params.get('learning_rate', lr)

grid_elapsed = time.time() - grid_start
print(f"\nGrid search 完成，最佳參數: {best_params}，交叉驗證準確率: {best_score:.4f}，耗時: {grid_elapsed:.2f} 秒")

# 以最佳參數建立 TabNet 模型，將 learning_rate 從 best_params 中取出
final_lr = best_params.pop('learning_rate')
optimizer_params = dict(lr=final_lr)
best_tabnet = TabNetClassifier(
    verbose=1,
    optimizer_params=optimizer_params,
    **best_params
)

# 訓練最佳 TabNet 模型（可視需求調整 max_epochs、patience 以及 batch_size）
print("\n開始訓練最佳 TabNet 模型...")
train_start = time.time()
best_tabnet.fit(
    X_train.values, y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
    drop_last=False
)
train_elapsed = time.time() - train_start
print(f"TabNet 模型訓練完成，耗時: {train_elapsed:.2f} 秒")

# 在驗證集上評估
y_pred_val = best_tabnet.predict(X_val.values)
print("\nTabNet 驗證集 Accuracy: {:.4f}".format(accuracy_score(y_val, y_pred_val)))
print("TabNet 驗證集 Classification Report:\n", classification_report(y_val, y_pred_val))

# ----- 預測測試集並輸出結果 -----
test_pred = best_tabnet.predict(X_test.values)
# 將預測結果由 0,1 轉回原始 1,2 格式
test_pred = test_pred + 1
if 'id' in test.columns:
    submission = pd.DataFrame({'id': test['id'], 'gender': test_pred})
else:
    submission = pd.DataFrame({'id': test.index, 'gender': test_pred})
submission.to_csv('submission.csv', index=False)
print("預測結果已儲存至 submission.csv")
