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

# 若尚未下載 nltk 資源，可取消下列註解執行一次
# nltk.download('stopwords')
# nltk.download('punkt')

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
    
    # 7. 拼寫檢查與修正（注意：此步驟可能會耗時且不一定對所有應用都有幫助）
    try:
        from textblob import TextBlob
        text = str(TextBlob(text).correct())
    except Exception as e:
        pass  # 若失敗則略過此步驟

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

# 1. 資料讀取（讀取原始資料）
train_df = pd.read_csv(r'D:\datascience\Sec_homework\output_gain_gender_sep.csv')
test_df = pd.read_csv(r'D:\datascience\Sec_homework\output_gain_all.csv')

# 用空字串填補 self_intro 欄位的缺失值
train_df['self_intro'] = train_df['self_intro'].fillna("")
test_df['self_intro'] = test_df['self_intro'].fillna("")

# 進行文字前處理，使用整合的步驟
print("進行文字前處理...")
train_df['self_intro'] = train_df['self_intro'].apply(preprocess_text)
test_df['self_intro'] = test_df['self_intro'].apply(preprocess_text)

num_cols = ['height', 'weight', 'iq', 'fb_friends', 'yt']
text_cols = ['self_intro']

# 將 gender 轉換為二元標籤：原 1 (男) 變 0，2 (女) 變 1
train_df['gender_bin'] = train_df['gender'].apply(lambda x: 0 if x == 1 else 1)

# 2. 文字特徵提取：採用 SentenceTransformer（使用 paraphrase-distilroberta-base-v1 模型）取得語意嵌入
model_name = "paraphrase-distilroberta-base-v1"  # 此處改用該模型
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

embed_dim = train_embeddings.shape[1]
train_embed_df = pd.DataFrame(train_embeddings, columns=[f"embed_{i}" for i in range(embed_dim)], index=train_df.index)
test_embed_df = pd.DataFrame(test_embeddings, columns=[f"embed_{i}" for i in range(embed_dim)], index=test_df.index)

# 3. 整合數值特徵與文字嵌入（不進行額外的特徵縮放）
train_features = pd.concat([train_df[num_cols].reset_index(drop=True),
                            train_embed_df.reset_index(drop=True)], axis=1)
test_features = pd.concat([test_df[num_cols].reset_index(drop=True),
                           test_embed_df.reset_index(drop=True)], axis=1)

# 4. 切分訓練集與驗證集，並使用 SMOTE 平衡資料
X = train_features.copy()
y = train_df['gender_bin']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 5. 建立調參的模型及縮小範圍的參數網格（僅保留 XGB）
models_params = [
    ('XGB', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='gpu_hist', n_jobs=-1),
     {
         'n_estimators': [100, 300],
         'max_depth': [3, 7],
         'learning_rate': [0.05, 0.1],
         'subsample': [0.8, 1.0],
         'colsample_bytree': [0.8, 1.0],
         'gamma': [0, 0.1]
     })
]

results = {}
best_model = None
best_f1 = -np.inf

for name, model, params in models_params:
    print(f"\n進行 {name} 的 GridSearchCV 調參...")
    start_model_time = time.time()
    # XGB 模型只取前 150 個特徵
    X_train_model = X_train.iloc[:, :150]
    X_val_model = X_val.iloc[:, :150]
    grid = GridSearchCV(model, params, cv=5, scoring='f1', verbose=0, n_jobs=1)
    grid.fit(X_train_model, y_train)
    elapsed = time.time() - start_model_time
    best_estimator = grid.best_estimator_
    y_pred_val = best_estimator.predict(X_val_model)
    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    print(f"{name} 最佳參數: {grid.best_params_}")
    print(f"{name} 驗證集結果: Accuracy = {acc:.4f}, F1 = {f1:.4f}，耗時 {elapsed:.2f} 秒")
    results[name] = {'model': best_estimator, 'accuracy': acc, 'f1': f1}
    if f1 > best_f1:
        best_f1 = f1
        best_model = (name, best_estimator)

print("\n各模型比較結果：")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
print(f"\n選出的最佳模型: {best_model[0]} (F1 = {best_f1:.4f})")

# 6. 使用最佳模型對 test 資料進行預測
final_model = best_model[1]
test_pred = final_model.predict(test_features.iloc[:, :150])
test_pred_converted = test_pred + 1

if 'id' in test_df.columns:
    submission = pd.DataFrame({'id': test_df['id'], 'gender': test_pred_converted})
else:
    submission = pd.DataFrame({'id': test_df.index, 'gender': test_pred_converted})
submission.to_csv('submission.csv', index=False)
print("\n預測結果已儲存至 submission.csv")
