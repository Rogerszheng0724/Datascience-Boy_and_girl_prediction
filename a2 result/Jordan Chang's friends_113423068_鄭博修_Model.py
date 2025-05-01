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
from sentence_transformers import SentenceTransformer  # For semantic embeddings
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Additional imports for model comparison
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Set English stopwords
english_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # 3. Remove URLs
    text = re.sub(r'http\S+', '', text)
    # 4. Replace digits with "NUM"
    text = re.sub(r'\d+', 'NUM', text)
    # 5. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # 6. Remove extra whitespace and control characters
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. (Optional) Spell correction (can be time-consuming)
    try:
        from textblob import TextBlob
        text = str(TextBlob(text).correct())
    except Exception as e:
        pass

    # 8. Expand abbreviations
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
    
    # 9. Remove stopwords
    tokens = [token for token in tokens if token.lower() not in english_stopwords]
    
    return " ".join(tokens)

# ------------------ Data Loading and Preprocessing ------------------
# Load only the necessary columns
train_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\train.csv', usecols=['height', 'weight', 'self_intro', 'gender'])
test_df = pd.read_csv(r'D:\datascience\Sec_homework\dataset\test.csv', usecols=['height', 'weight', 'self_intro'])

# Fill missing self_intro values
train_df['self_intro'] = train_df['self_intro'].fillna("")
test_df['self_intro'] = test_df['self_intro'].fillna("")

print("Performing text preprocessing...")
train_df['self_intro'] = train_df['self_intro'].apply(preprocess_text)
test_df['self_intro'] = test_df['self_intro'].apply(preprocess_text)

# Only keep numerical features: height and weight; text column: self_intro
num_cols = ['height', 'weight']

# Convert gender to binary labels: assume 1 (male) becomes 0, 2 (female) becomes 1
train_df['gender_bin'] = train_df['gender'].apply(lambda x: 0 if x == 1 else 1)

# ------------------ Obtain Semantic Embeddings ------------------
model_name = "all-distilroberta-v1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(model_name, device=device)
print("SentenceTransformer is using device:", embedder.device)

print("Computing embeddings for train and test self_intro...")
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
print(f"Embedding completed in {time.time() - start_time:.2f} seconds")

# Convert embeddings to DataFrame
embed_dim = train_embeddings.shape[1]
train_embed_df = pd.DataFrame(train_embeddings, columns=[f"embed_{i}" for i in range(embed_dim)], index=train_df.index)
test_embed_df = pd.DataFrame(test_embeddings, columns=[f"embed_{i}" for i in range(embed_dim)], index=test_df.index)

# Combine numerical features with text embeddings
train_features = pd.concat([train_df[num_cols].reset_index(drop=True),
                            train_embed_df.reset_index(drop=True)], axis=1)
test_features = pd.concat([test_df[num_cols].reset_index(drop=True),
                           test_embed_df.reset_index(drop=True)], axis=1)

# ------------------ Train-Test Split and SMOTE ------------------
X = train_features.copy()
y = train_df['gender_bin']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Use SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ------------------ Model Training with XGBoost via GridSearchCV ------------------
# Build a pipeline: feature selection (SelectKBest) then XGBoost classifier
pipeline = Pipeline([
    ('select', SelectKBest(score_func=f_classif, k=150)),
    ('clf', XGBClassifier(eval_metric='logloss', random_state=42, tree_method='gpu_hist', n_jobs=-1))
])
# Parameter grid for GridSearchCV (note the step names)
param_grid = {
    'clf__n_estimators': [100, 300],
    'clf__max_depth': [3, 7],
    'clf__learning_rate': [0.05, 0.1],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0],
    'clf__gamma': [0, 0.1]
}

print("\nPerforming GridSearchCV for XGBoost pipeline...")
start_model_time = time.time()
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=0, n_jobs=1)
grid.fit(X_train, y_train)
elapsed = time.time() - start_model_time

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
y_pred_val = best_model.predict(X_val)
acc = accuracy_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)
print("Validation Results: Accuracy = {:.4f}, F1 = {:.4f}, Training Time = {:.2f} seconds".format(acc, f1, elapsed))

# ------------------ Save Predictions ------------------
# Use the best model to predict test data
test_pred = best_model.predict(test_features)
# If original gender labels are 1 and 2, add 1 back
test_pred_converted = test_pred + 1

if 'id' in test_df.columns:
    submission = pd.DataFrame({'id': test_df['id'], 'gender': test_pred_converted})
else:
    submission = pd.DataFrame({'id': test_df.index, 'gender': test_pred_converted})
submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to submission.csv")

# =============================================================================
# 5. XGBoost Feature Importance Plot
# =============================================================================
# Retrieve the indices of the selected features from SelectKBest
selected_indices = best_model.named_steps['select'].get_support(indices=True)
selected_feature_names = X_train.columns[selected_indices]
importance_scores = best_model.named_steps['clf'].feature_importances_

# Create a DataFrame of feature importance scores
feature_importance_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance Score': importance_scores
})
# Sort and take the top 20 features
top20_features = feature_importance_df.sort_values(by='Importance Score', ascending=False).head(20)

plt.figure(figsize=(10, 6))
plt.barh(top20_features['Feature'], top20_features['Importance Score'])
plt.xlabel('Importance Score')
plt.title('Top 20 Feature Importance from XGBoost')
plt.gca().invert_yaxis()  # Highest importance on top
plt.tight_layout()
plt.show()

# =============================================================================
# 6. Model Comparison: Random Forest, XGBoost, and MLP
# =============================================================================
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, tree_method='gpu_hist', n_jobs=-1),
    'MLP': MLPClassifier(random_state=42, max_iter=200)
}

comparison_results = []
for model_name, model_instance in models.items():
    pipeline_model = Pipeline([
        ('select', SelectKBest(score_func=f_classif, k=150)),
        ('clf', model_instance)
    ])
    start_time_model = time.time()
    pipeline_model.fit(X_train, y_train)
    training_time = time.time() - start_time_model
    y_val_pred = pipeline_model.predict(X_val)
    acc_model = accuracy_score(y_val, y_val_pred)
    f1_model = f1_score(y_val, y_val_pred)
    comparison_results.append({
        'Model': model_name,
        'Accuracy': acc_model,
        'F1 Score': f1_model,
        'Training Time (s)': training_time
    })

comparison_df = pd.DataFrame(comparison_results)
print("\nModel Comparison Results:")
print(comparison_df)

# Plot Accuracy and F1 Score as a grouped bar chart
import numpy as np
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
indices = np.arange(len(comparison_df))
ax.bar(indices, comparison_df['Accuracy'], bar_width, label='Accuracy')
ax.bar(indices + bar_width, comparison_df['F1 Score'], bar_width, label='F1 Score')
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Comparison: Accuracy and F1 Score')
ax.set_xticks(indices + bar_width / 2)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
plt.tight_layout()
plt.show()

# Plot Training Time as a separate bar chart
plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['Training Time (s)'])
plt.ylabel('Training Time (s)')
plt.title('Model Comparison: Training Time')
plt.tight_layout()
plt.show()
