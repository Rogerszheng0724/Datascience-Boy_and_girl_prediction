import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # 改為 SMOTE

# ------------------ 輔助函數 ------------------
def scale_features_by_importance(df, imp_dict):
    df_scaled = df.copy()
    for c in df_scaled.columns:
        if c in imp_dict:
            df_scaled[c] = df_scaled[c] * imp_dict[c]
    return df_scaled

def get_feature_importance(estimator, feature_names):
    if hasattr(estimator, 'named_steps'):
        est = list(estimator.named_steps.values())[-1]
    else:
        est = estimator

    if hasattr(est, 'feature_importances_'):
        importances = est.feature_importances_
    elif hasattr(est, 'coef_'):
        importances = np.abs(est.coef_)[0]
    else:
        importances = np.ones(len(feature_names))
    return dict(zip(feature_names, importances))

# ------------------ 0. 定義利用 BERT 取得句向量的函數 ------------------
def get_bert_embeddings(texts, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            # 移至 GPU
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            outputs = model(**encoded_input)
            token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attention_mask = encoded_input['attention_mask']
            # 利用 attention mask 計算平均
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(sentence_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

# 設定 device (若有 GPU 可自動使用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 載入 BERT 模型與 tokenizer (此處以 bert-base-uncased 為例，可依需求更換模型)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.to(device)

# ------------------ 1. 資料讀取與前處理 ------------------
train_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\train.csv')
test_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\test.csv')

# 填補數值與文字欄位的缺失值
num_cols = ['height', 'weight', 'iq', 'fb_friends', 'yt']
text_cols = ['self_intro']
for col in num_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
    test_df[col].fillna(test_df[col].mean(), inplace=True)
for col in text_cols:
    train_df[col].fillna("", inplace=True)
    test_df[col].fillna("", inplace=True)

# ※ 已刪除離群值處理相關程式碼

# 將 gender 轉換為二元標籤 (1→男, 2→女) ：0=男, 1=女
train_df['gender_bin'] = train_df['gender'].apply(lambda x: 0 if x == 1 else 1)

# 數值欄位正規化
scaler_for_features = StandardScaler()
train_df[num_cols] = scaler_for_features.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler_for_features.transform(test_df[num_cols])

# ------------------ 2. 特徵工程 - 文字資料使用 BERT 取得語意嵌入 ------------------
# 利用 BERT 取得 self_intro 欄位句向量
train_texts = train_df['self_intro'].tolist()
test_texts = test_df['self_intro'].tolist()

print("取得 train text embeddings ...")
train_embeddings = get_bert_embeddings(train_texts, tokenizer, bert_model, device, batch_size=32)
print("取得 test text embeddings ...")
test_embeddings = get_bert_embeddings(test_texts, tokenizer, bert_model, device, batch_size=32)
embed_dim = train_embeddings.shape[1]

# 轉換成 DataFrame 並命名各向量維度
train_embed_df = pd.DataFrame(train_embeddings, columns=[f"embed_{i}" for i in range(embed_dim)], index=train_df.index)
test_embed_df = pd.DataFrame(test_embeddings, columns=[f"embed_{i}" for i in range(embed_dim)], index=test_df.index)

# 整合數值特徵與 self_intro 的語意嵌入
feature_cols = ['height', 'weight', 'iq', 'fb_friends', 'yt']
train_features = pd.concat([train_df[feature_cols].reset_index(drop=True),
                            train_embed_df.reset_index(drop=True)], axis=1)
test_features = pd.concat([test_df[feature_cols].reset_index(drop=True),
                           test_embed_df.reset_index(drop=True)], axis=1)

# ------------------ 3. 切分訓練與驗證集 ------------------
X = train_features.copy()
y = train_df['gender_bin']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 使用 SMOTE 過採樣 (原 ADASYN 改成 SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ------------------ 4. 定義模型與參數網格 ------------------
models_params = [
    ('LR', Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        {'lr__C': [0.01, 0.1, 1, 10, 100]}
    ),
    ('RF', RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
    ),
    ('GB', GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    ),
    ('SVC', Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ]),
        {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto']}
    ),
    ('KNN', Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ]),
        {'knn__n_neighbors': [3, 5, 7]}
    ),
    ('DT', DecisionTreeClassifier(random_state=42),
        {'max_depth': [None, 5, 10, 20]}
    ),
    ('XGB', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
    )
]

# ------------------ 5. 兩階段訓練 ------------------
retrained_models = {}

print("開始 GridSearchCV 調參與兩階段訓練 (取得重要度 → 縮放特徵 → 再訓練)...")
for name, model, params in models_params:
    print(f"\n調參 {name} ...")
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    print(f"{name} 最佳參數: {grid.best_params_}")
    
    imp_dict = get_feature_importance(best_estimator, X_train.columns)
    X_train_scaled = scale_features_by_importance(X_train, imp_dict)
    X_val_scaled = scale_features_by_importance(X_val, imp_dict)
    
    if name == 'LR':
        C_val = grid.best_params_.get('lr__C', 1.0)
        new_model = LogisticRegression(max_iter=1000, random_state=42, C=C_val)
    elif name == 'SVC':
        C_val = grid.best_params_.get('svc__C', 1.0)
        gamma_val = grid.best_params_.get('svc__gamma', 'scale')
        new_model = SVC(probability=True, random_state=42, C=C_val, gamma=gamma_val)
    elif name == 'KNN':
        n_neighbors_val = grid.best_params_.get('knn__n_neighbors', 5)
        new_model = KNeighborsClassifier(n_neighbors=n_neighbors_val)
    elif name == 'RF':
        new_model = RandomForestClassifier(random_state=42, **grid.best_params_)
    elif name == 'GB':
        new_model = GradientBoostingClassifier(random_state=42, **grid.best_params_)
    elif name == 'DT':
        new_model = DecisionTreeClassifier(random_state=42, **grid.best_params_)
    elif name == 'XGB':
        new_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **grid.best_params_)
    else:
        new_model = best_estimator

    new_model.fit(X_train_scaled, y_train)
    y_pred = new_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    print(f"{name} (重新訓練後) 驗證集表現: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    retrained_models[name] = (new_model, imp_dict, metrics)

# ------------------ 6. 軟投票集成 ------------------
probs_ensemble = None
n_models = len(retrained_models)
for name, (model, imp_dict, _) in retrained_models.items():
    X_val_scaled = scale_features_by_importance(X_val, imp_dict)
    prob = model.predict_proba(X_val_scaled)
    probs_ensemble = prob if probs_ensemble is None else probs_ensemble + prob
probs_ensemble /= n_models
y_pred_ensemble = np.argmax(probs_ensemble, axis=1)
ens_acc = accuracy_score(y_val, y_pred_ensemble)
ens_prec = precision_score(y_val, y_pred_ensemble)
ens_rec = recall_score(y_val, y_pred_ensemble)
ens_f1 = f1_score(y_val, y_pred_ensemble)
ensemble_metrics = {'Accuracy': ens_acc, 'Precision': ens_prec, 'Recall': ens_rec, 'F1': ens_f1}
print("\nEnsemble (軟投票) 驗證集表現:")
print(f"Accuracy={ens_acc:.4f}, Precision={ens_prec:.4f}, Recall={ens_rec:.4f}, F1={ens_f1:.4f}")

# ------------------ 7. 統整結果與最佳模型 ------------------
best_method = None
best_acc = 0
print("\n各模型驗證集評估結果 (重新訓練後):")
for name, (model, imp_dict, metrics) in retrained_models.items():
    print(f"{name}: Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")
    if metrics['Accuracy'] > best_acc:
        best_acc = metrics['Accuracy']
        best_method = name
print(f"Ensemble: Accuracy={ensemble_metrics['Accuracy']:.4f}, Precision={ensemble_metrics['Precision']:.4f}, Recall={ensemble_metrics['Recall']:.4f}, F1={ensemble_metrics['F1']:.4f}")
if ensemble_metrics['Accuracy'] > best_acc:
    best_acc = ensemble_metrics['Accuracy']
    best_method = 'Ensemble'

print(f"\n最佳模型為: {best_method}，驗證集 Accuracy={best_acc:.4f}")

# ------------------ 8. 測試集預測與輸出 submission ------------------
if best_method == 'Ensemble':
    probs_test = None
    for name, (model, imp_dict, _) in retrained_models.items():
        X_test_scaled = scale_features_by_importance(test_features, imp_dict)
        prob = model.predict_proba(X_test_scaled)
        probs_test = prob if probs_test is None else probs_test + prob
    probs_test /= n_models
    test_pred = np.argmax(probs_test, axis=1)
else:
    model, imp_dict, _ = retrained_models[best_method]
    X_test_scaled = scale_features_by_importance(test_features, imp_dict)
    test_pred = model.predict(X_test_scaled)

# 將 0,1 轉回原本的 1,2
test_pred_converted = test_pred + 1
if 'id' in test_df.columns:
    submission = pd.DataFrame({'id': test_df['id'], 'gender': test_pred_converted})
else:
    submission = pd.DataFrame({'id': test_df.index, 'gender': test_pred_converted})
submission.to_csv('submission.csv', index=False)
print("\n預測結果已儲存至 submission.csv")
