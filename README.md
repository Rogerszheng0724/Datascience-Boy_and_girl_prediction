# Boy or Girl Prediction 2025

[![banner](https://ppt.cc/f8NsTx)](https://www.kaggle.com/competitions/boy-or-girl-2025-new)

本專案參加 Kaggle 比賽：[Boy or Girl 2025](https://www.kaggle.com/competitions/boy-or-girl-2025-new)，目標是透過 **結構化資料 + 自我介紹文字** 來預測性別。我們針對 **資料前處理、特徵工程與分類模型** 建立了一套完整流程。

---

## 📂 專案結構
- `data/` 原始與處理後資料
- `notebooks/` 資料分析與模型實驗
- `src/` 主程式碼
- `models/` 訓練好的模型
- `README.md` 專案說明文件

---

## 🚀 方法論

### 1. 資料前處理
- **缺失值補值 (Imputation)**  
  - 測試方法：Mean, Median, KNN, EM, GAIN  
  - **最佳結果**：GAIN（特別在 height 特徵上效果最佳）【圖片，p.5 Graph 1】

- **異常值處理 (Outlier Treatment)**  
  - 使用 IQR (Interquartile Range)  
  - 閾值 = 0.2 IQR → 保留合理極端值，排除輸入錯誤值  
  - 【圖片，p.6 Graph 2】

- **文字欄位 (self_intro)**  
  - 預處理：去除停用詞、縮寫展開、標點清理、大小寫統一、拼字校正  
  - 向量化方式比較：TF-IDF / Word2Vec / Sentence-BERT (SBERT)  
  - **最佳結果**：`all-distilroberta-v1`:contentReference[oaicite:1]{index=1}

- **類別不平衡處理**  
  - 採用 SMOTE 過採樣平衡資料

---

### 2. 特徵工程
- **特徵選擇**  
  - 數值變數 vs. 二元類別 → Point-biserial correlation  
  - 類別變數 vs. 類別 → Cramér’s V  
  - **選出高度相關特徵**：Height (r = -0.571), Weight (r = -0.404)  
  - 【圖片，p.14 Graph 3】

- **正規化**  
  - 嘗試 Z-score normalization  
  - 結果：對樹模型無顯著提升，最終未採用

- **降維**  
  - PCA, Autoencoder → 減少維度  
  - 結果：表現下降，最終未採用

---

### 3. 模型
- **嘗試方法**  
  - Machine Learning: Random Forest, XGBoost  
  - Deep Learning: MLP  

- **超參數調整**  
  - GridSearchCV, Bayesian Optimization  
  - Cross-validation: 5-fold  

- **最佳組合**  
  - **XGBoost** + 前 150 個最重要特徵  
  - Validation split = 0.3 → 最佳泛化效果  
  - 【圖片，p.20 Graph 4】

---

## 📊 結果
- XGBoost + GAIN + SBERT (all-distilroberta-v1) → **最佳整體表現**
- 強化模型泛化能力，避免過度依賴少數異常值
- Deep Learning 模型在小資料集上效果不佳，僅作為對照

---

## 🧩 工作分工
| 成員 | 貢獻 |
|------|------|
| Rogers Zheng | 實驗設計、文獻回顧、缺失值補值、模型訓練、文件撰寫 (50%) |
| Brian Zou   | 實驗設計、文獻回顧、模型訓練、文件撰寫 (50%) |

---

## 📌 參考文獻
部分核心參考文獻：
- Yoon et al. (2018). **GAIN: Missing Data Imputation using Generative Adversarial Nets**. arXiv:1806.02920  
- Reimers & Gurevych (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. EMNLP-IJCNLP  
- Breiman (2001). **Random Forests**. Machine Learning  

完整文獻請見原始文件:contentReference[oaicite:2]{index=2}

---

## 📎 比賽連結
👉 [Kaggle: Boy or Girl 2025](https://www.kaggle.com/competitions/boy-or-girl-2025-new)
