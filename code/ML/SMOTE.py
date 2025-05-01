import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
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

# 新增 SMOTE 匯入
from imblearn.over_sampling import SMOTE

# ------------------ 輔助函數 ------------------
def scale_features_by_importance(df, imp_dict):
    """
    根據 importance 字典，將 DataFrame 中每個欄位乘上對應的重要度。
    """
    df_scaled = df.copy()
    for c in df_scaled.columns:
        if c in imp_dict:
            df_scaled[c] = df_scaled[c] * imp_dict[c]
    return df_scaled

def get_feature_importance(estimator, feature_names):
    """
    從模型中取得特徵重要度：
      - 若模型有 feature_importances_ 屬性則直接使用，
      - 若有 coef_ 屬性則取絕對值（適用於 LR），
      - 否則回傳全 1 向量。
    若 estimator 是 Pipeline，則從最後一層提取。
    回傳 dict {feature_name: importance_value}
    """
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

# ------------------ 1. 資料讀取與前處理 ------------------
train_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\train.csv')
test_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\test.csv')

# (a) 補缺：數值型以平均值，文字型以空字串
num_cols = ['height', 'weight', 'iq', 'fb_friends', 'yt']
text_cols = ['self_intro']
for col in num_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
    test_df[col].fillna(test_df[col].mean(), inplace=True)
for col in text_cols:
    train_df[col].fillna("", inplace=True)
    test_df[col].fillna("", inplace=True)

# (b) 處理星座：補缺並轉換（支援英文與中文）
train_df['star_sign'].fillna(train_df['star_sign'].mode()[0], inplace=True)
test_df['star_sign'].fillna(test_df['star_sign'].mode()[0], inplace=True)
star_sign_map = {
    'Aries': 1, 'Taurus': 2, 'Gemini': 3, 'Cancer': 4,
    'Leo': 5, 'Virgo': 6, 'Libra': 7, 'Scorpio': 8,
    'Sagittarius': 9, 'Capricorn': 10, 'Aquarius': 11, 'Pisces': 12,
    '牡羊座': 1, '金牛座': 2, '雙子座': 3, '巨蟹座': 4,
    '獅子座': 5, '處女座': 6, '天秤座': 7, '天蠍座': 8,
    '射手座': 9, '摩羯座': 10, '水瓶座': 11, '雙魚座': 12
}
if train_df['star_sign'].dtype == object:
    train_df['star_sign'] = train_df['star_sign'].map(star_sign_map)
if test_df['star_sign'].dtype == object:
    test_df['star_sign'] = test_df['star_sign'].map(star_sign_map)

# (c) 轉換 gender (1→男, 2→女) 為二元標籤：0=男, 1=女
train_df['gender_bin'] = train_df['gender'].apply(lambda x: 0 if x==1 else 1)

# (d) 離群值處理：以 IQR 對數值型資料做上下限截斷
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)
for col in num_cols:
    train_df[col] = cap_outliers(train_df[col])
    test_df[col] = cap_outliers(test_df[col])

# ------------------ 2. 特徵工程 ------------------
# (a) 對 self_intro 文字欄位進行 TF-IDF 向量化
tfidf = TfidfVectorizer(max_features=100)
tfidf_train = tfidf.fit_transform(train_df['self_intro'].astype(str))
tfidf_test = tfidf.transform(test_df['self_intro'].astype(str))
tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), 
                              columns=[f"tfidf_{i}" for i in range(tfidf_train.shape[1])],
                              index=train_df.index)
tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), 
                             columns=[f"tfidf_{i}" for i in range(tfidf_test.shape[1])],
                             index=test_df.index)
# (b) 合併數值特徵與 TF-IDF 特徵；原始特徵：height, weight, iq, fb_friends, yt, star_sign
feature_cols = ['height', 'weight', 'iq', 'fb_friends', 'yt', 'star_sign']
train_features = pd.concat([train_df[feature_cols].reset_index(drop=True),
                            tfidf_train_df.reset_index(drop=True)], axis=1)
test_features = pd.concat([test_df[feature_cols].reset_index(drop=True),
                           tfidf_test_df.reset_index(drop=True)], axis=1)
# (c) 若星座為數值，進行 one-hot encoding
train_features = pd.get_dummies(train_features, columns=['star_sign'], drop_first=True)
test_features = pd.get_dummies(test_features, columns=['star_sign'], drop_first=True)
train_features, test_features = train_features.align(test_features, join='left', axis=1, fill_value=0)

# ------------------ 3. 切分訓練與驗證集 ------------------
X = train_features.copy()
y = train_df['gender_bin']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)

# 新增 SMOTE 過採樣：針對少數類別（原始 gender 為 2，即 gender_bin==1）進行過採樣
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

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

# ------------------ 5. 兩階段訓練：調參 → 取得重要度 → 縮放特徵 → 重新訓練 ------------------
# 用 retrained_models 存放每支模型的 (重新訓練後的模型, importance 字典, 評估指標)
retrained_models = {}

print("開始 GridSearchCV 調參與兩階段訓練 (取得重要度 → 縮放特徵 → 再訓練)...")
for name, model, params in models_params:
    print(f"\n調參 {name} ...")
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    print(f"{name} 最佳參數: {grid.best_params_}")
    
    # 取得重要度：利用原始 X_train 的欄位名稱
    imp_dict = get_feature_importance(best_estimator, X_train.columns)
    
    # 利用重要度縮放訓練與驗證資料
    X_train_scaled = scale_features_by_importance(X_train, imp_dict)
    X_val_scaled = scale_features_by_importance(X_val, imp_dict)
    
    # 根據最佳參數，重新建立不含 pipeline 的模型
    if name == 'LR':
        best_params = grid.best_params_
        C_val = best_params.get('lr__C', 1.0)
        new_model = LogisticRegression(max_iter=1000, random_state=42, C=C_val)
    elif name == 'SVC':
        best_params = grid.best_params_
        C_val = best_params.get('svc__C', 1.0)
        gamma_val = best_params.get('svc__gamma', 'scale')
        new_model = SVC(probability=True, random_state=42, C=C_val, gamma=gamma_val)
    elif name == 'KNN':
        best_params = grid.best_params_
        n_neighbors_val = best_params.get('knn__n_neighbors', 5)
        new_model = KNeighborsClassifier(n_neighbors=n_neighbors_val)
    elif name == 'RF':
        best_params = grid.best_params_
        n_estimators_val = best_params.get('n_estimators', 100)
        max_depth_val = best_params.get('max_depth', None)
        new_model = RandomForestClassifier(random_state=42, n_estimators=n_estimators_val, max_depth=max_depth_val)
    elif name == 'GB':
        best_params = grid.best_params_
        n_estimators_val = best_params.get('n_estimators', 100)
        learning_rate_val = best_params.get('learning_rate', 0.1)
        new_model = GradientBoostingClassifier(random_state=42, n_estimators=n_estimators_val, learning_rate=learning_rate_val)
    elif name == 'DT':
        best_params = grid.best_params_
        max_depth_val = best_params.get('max_depth', None)
        new_model = DecisionTreeClassifier(random_state=42, max_depth=max_depth_val)
    elif name == 'XGB':
        best_params = grid.best_params_
        n_estimators_val = best_params.get('n_estimators', 100)
        max_depth_val = best_params.get('max_depth', 3)
        new_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                                  n_estimators=n_estimators_val, max_depth=max_depth_val)
    else:
        new_model = best_estimator  # fallback

    # 重新訓練新模型（在縮放後的資料上）
    new_model.fit(X_train_scaled, y_train)
    y_pred = new_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    print(f"{name} (重新訓練後) 驗證集表現: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    retrained_models[name] = (new_model, imp_dict, metrics)

# ------------------ 6. 軟投票集成 (Ensemble) ------------------
# 針對每個模型，先對驗證集依其重要度縮放，再取 predict_proba，平均後取最大機率類別
probs_ensemble = None
n_models = len(retrained_models)
for name, (model, imp_dict, metrics) in retrained_models.items():
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

# ------------------ 7. 統整結果並選出最佳模型 ------------------
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
    # ensemble 需各模型對測試資料分別依其重要度縮放後取機率再平均
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
