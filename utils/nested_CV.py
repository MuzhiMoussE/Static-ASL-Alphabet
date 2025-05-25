import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import numpy as np

def save_cv_results(model_scores, output_dir="results"):

    os.makedirs(output_dir, exist_ok=True)

    # 转换为 DataFrame
    df = pd.DataFrame([
        {"Model": name, "CV Accuracy": score[0], "Std Dev": score[1]}
        for name, score in model_scores.items()
    ])
    df_sorted = df.sort_values("CV Accuracy", ascending=False)
    csv_path = os.path.join(output_dir, "cv_results.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"Cross-validation results saved to {csv_path}")

    # 创建图像：柱状图 + 表格
    fig, ax = plt.subplots(nrows=2, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(hspace=0.3)

    # 上：柱状图
    ax[0].bar(df_sorted["Model"], df_sorted["CV Accuracy"], yerr=df_sorted["Std Dev"],
              capsize=5, color="skyblue")
    ax[0].set_title("Model Comparison via Nested Cross-Validation")
    ax[0].set_ylabel("CV Accuracy")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 下：表格
    table_data = [["Model", "CV Accuracy", "Std Dev"]] + \
                 [[row["Model"], f"{row['CV Accuracy']:.4f}", f"{row['Std Dev']:.4f}"]
                  for _, row in df_sorted.iterrows()]

    ax[1].axis("off")
    table = ax[1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.4)

    # 保存
    output_path = os.path.join(output_dir, "cv_results_combined.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined CV result figure saved to {output_path}")




def run_nested_cv_all_models(X, y):
    print("\nRunning Nested Cross-Validation...")

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}

    # --- 1. KNN ---
    print("\n[KNN] Cross-validating...")
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=inner_cv, n_jobs=-1)
    knn_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=5, desc="KNN Outer CV"):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        knn_grid.fit(X_train, y_train)
        knn_scores.append(knn_grid.score(X_val, y_val))
    knn_scores = np.array(knn_scores)
    print(f"KNN: Accuracy = {knn_scores.mean():.4f} ± {knn_scores.std():.4f}")
    results["KNN"] = (knn_scores.mean(), knn_scores.std())

    # --- 2. SVM ---
    print("\n[SVM] Cross-validating...")
    svm_params = {'C': [0.01, 0.1, 1, 10]}
    svm_grid = GridSearchCV(SVC(), svm_params, cv=inner_cv, n_jobs=-1)
    svm_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=5, desc="SVM Outer CV"):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        svm_grid.fit(X_train, y_train)
        svm_scores.append(svm_grid.score(X_val, y_val))
    svm_scores = np.array(svm_scores)
    print(f"SVM: Accuracy = {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
    results["SVM"] = (svm_scores.mean(), svm_scores.std())

    # --- 3. Naive Bayes ---
    print("\n[Naive Bayes] Cross-validating...")
    nb_params = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
    nb_grid = GridSearchCV(GaussianNB(), nb_params, cv=inner_cv, n_jobs=-1)
    nb_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=5, desc="NB Outer CV"):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        nb_grid.fit(X_train, y_train)
        nb_scores.append(nb_grid.score(X_val, y_val))
    nb_scores = np.array(nb_scores)
    print(f"Naive Bayes: Accuracy = {nb_scores.mean():.4f} ± {nb_scores.std():.4f}")
    results["Naive Bayes"] = (nb_scores.mean(), nb_scores.std())
    save_cv_results(results)

    return results
