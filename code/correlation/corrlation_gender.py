import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pointbiserialr, chi2_contingency

# ------------------ 讀取資料 ------------------
df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\no_sep.csv')

# 將 gender 轉換為二元變數 (0: 男生, 1: 女生)
df['gender_bin'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

# ------------------ 連續變數與類別變數相關性分析 (原始方法) ------------------
# 連續變數：使用點二列相關 (這邊針對其他數值變數)
continuous_vars = ['height', 'weight', 'iq', 'fb_friends', 'yt']
pb_results = {}
for var in continuous_vars:
    valid = df[['gender_bin', var]].dropna()
    corr, p_value = pointbiserialr(valid[var], valid['gender_bin'])
    pb_results[var] = (corr, p_value)

print("連續變數 (點二列相關) 與 gender 的相關性:")
for var, (corr, p_value) in pb_results.items():
    print(f"  {var}: correlation = {corr:.3f}, p-value = {p_value:.3g}")

# 類別變數：使用卡方檢定後計算 Cramér's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

categorical_vars = ['star_sign', 'phone_os', 'sleepiness']
cramers_results = {}
for var in categorical_vars:
    contingency = pd.crosstab(df['gender'], df[var])
    v = cramers_v(contingency)
    cramers_results[var] = v

print("\n類別變數 (Cramér's V) 與 gender 的相關性:")
for var, v in cramers_results.items():
    print(f"  {var}: Cramér's V = {v:.3f}")

# ------------------ 使用 BERT 分析 self_intro 與 gender 的關係 ------------------
# 定義利用 BERT 取得句向量的函數 (平均池化)
def get_bert_embeddings(texts, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # 取得最後一層 hidden states (batch, seq_len, hidden_size)
            token_embeddings = outputs.last_hidden_state  
            attention_mask = encoded['attention_mask']
            # 平均池化 (忽略 padding)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(sentence_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

# 設定 device (若有 GPU 可自動使用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 載入 BERT 模型與 tokenizer (此處以 bert-base-uncased 為例，可依需求更換模型)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.to(device)

# 取得 self_intro 欄位的句向量 (BERT embedding)
texts = df['self_intro'].astype(str).tolist()
print("\n取得 self_intro 的 BERT embeddings ...")
embeddings = get_bert_embeddings(texts, tokenizer, model, device, batch_size=32)
embed_dim = embeddings.shape[1]
print(f"BERT embedding 維度: {embed_dim}")

# 計算每個 embedding 維度與 gender 的點二列相關性
bert_corr_results = []
for i in range(embed_dim):
    corr, p_value = pointbiserialr(embeddings[:, i], df['gender_bin'])
    bert_corr_results.append({'dimension': i, 'correlation': corr, 'p_value': p_value})

bert_corr_df = pd.DataFrame(bert_corr_results)
print("\nBERT self_intro 各維度與 gender 的點二列相關性:")
print(bert_corr_df.head(10))  # 先顯示前 10 個維度

# 若想檢視所有維度，取消下面的註解
# print(bert_corr_df)

# 可進一步統計平均相關程度
mean_abs_corr = bert_corr_df['correlation'].abs().mean()
print(f"\nBERT embedding 各維度絕對相關性的平均值: {mean_abs_corr:.3f}")
