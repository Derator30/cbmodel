import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv')
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df[['review']], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

train_pool = Pool(data=X_train, label=y_train, text_features=['review'])
test_pool = Pool(data=X_test, label=y_test, text_features=['review'])

hyperparams = [
    {
        'iterations': 300,
        'learning_rate': 0.1,
        'depth': 4,
        'l2_leaf_reg': 1
    },
    {
        'iterations': 300,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 1
    },
    {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 5,
        'l2_leaf_reg': 3
    },
    {
        'iterations': 500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 5
    },
    {
        'iterations': 700,
        'learning_rate': 0.02,
        'depth': 7,
        'l2_leaf_reg': 5
    },
    {
        'iterations': 700,
        'learning_rate': 0.01,
        'depth': 8,
        'l2_leaf_reg': 10
    }
]

results = []

for i, params in enumerate(hyperparams):
    print(f"\n{'=' * 50}")
    print(f"Обучение {i + 1}/{len(hyperparams)}")
    print(
        f"Параметры: iterations={params['iterations']}, lr={params['learning_rate']}, depth={params['depth']}, l2={params['l2_leaf_reg']}")
    print(f"{'=' * 50}")

    model = CatBoostClassifier(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params.get('l2_leaf_reg', 3),
        loss_function='Logloss',
        eval_metric='AUC',
        custom_metric=['Accuracy', 'Precision', 'Recall', 'AUC'],
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
        use_best_model=True
    )

    model.fit(train_pool, eval_set=test_pool, verbose=False)

    best_auc = model.best_score_['validation']['AUC']

    results.append({
        'id': i + 1,
        'params': params,
        'best_auc': best_auc,
        'model': model
    })

    print(f"Лучший AUC на валидации: {best_auc:.4f}")

print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ")
print(f"{'=' * 60}")

results_sorted = sorted(results, key=lambda x: x['best_auc'], reverse=True)

for r in results_sorted:
    print(
        f"Модель {r['id']}: AUC = {r['best_auc']:.4f} | params: iterations={r['params']['iterations']}, lr={r['params']['learning_rate']}, depth={r['params']['depth']}")

print(f"\n{'=' * 60}")
print(f"ЛУЧШАЯ МОДЕЛЬ: Модель {results_sorted[0]['id']}")
print(f"AUC: {results_sorted[0]['best_auc']:.4f}")
print(f"Параметры: {results_sorted[0]['params']}")
print(f"{'=' * 60}")

best_model = results_sorted[0]['model']
best_model.save_model('sentiment_model_best.cbm')
print("\nЛучшая модель сохранена как 'sentiment_model_best.cbm'")

history = best_model.get_evals_result()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'AUC', 'Logloss']

for i, metric in enumerate(metrics):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    if metric in history['learn']:
        ax.plot(history['learn'][metric], label=f'Train {metric}', linewidth=2)

    if metric in history['validation']:
        ax.plot(history['validation'][metric], label=f'Validation {metric}', linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} over Training (Best Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if metric != 'Logloss':
        ax.set_ylim(0, 1)

for i in range(len(metrics), 6):
    row, col = i // 3, i % 3
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 6))
model_ids = [r['id'] for r in results_sorted]
auc_scores = [r['best_auc'] for r in results_sorted]
colors = ['gold' if i == 0 else 'steelblue' for i in range(len(results_sorted))]
bars = ax2.bar([str(mid) for mid in model_ids], auc_scores, color=colors)
ax2.set_xlabel('Model ID')
ax2.set_ylabel('Best AUC')
ax2.set_title('Comparison of Hyperparameter Configurations')
ax2.set_ylim(0.85, 0.96)

for bar, auc in zip(bars, auc_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{auc:.4f}', ha='center', va='bottom',
             fontsize=10)

plt.tight_layout()
plt.show()

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train['review'])
X_test_tfidf = vectorizer.transform(X_test['review'])

lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

feature_names = vectorizer.get_feature_names_out()
importances = abs(lr.coef_[0])
top_idx = importances.argsort()[-20:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(20), importances[top_idx])
plt.yticks(range(20), feature_names[top_idx])
plt.xlabel('Coefficient Magnitude')
plt.title('Top-20 Important Words (Logistic Regression on TF-IDF)')
plt.tight_layout()
plt.show()