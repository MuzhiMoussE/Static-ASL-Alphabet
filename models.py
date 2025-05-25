import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

MODELS_DIR = "models"

def train_knn(X_train, y_train, n_neighbors=1):
    model_name = f"knn_model_n{n_neighbors}.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(model_path):
        print(f"Model {model_name} already exists. Loading...")
        return joblib.load(model_path)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("Training KNN...")
    knn.fit(X_train, y_train)

    joblib.dump(knn, model_path)  # Save model
    return knn

def train_svm(X_train, y_train, C=10, kernel='rbf'):
    model_name = f"svm_model_C{C}_kernel{kernel}.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(model_path):
        print(f"Model {model_name} already exists. Loading...")
        return joblib.load(model_path)

    svm = SVC(C=C, kernel=kernel, gamma='scale')
    print("Training SVM...")

    svm.fit(X_train, y_train)

    joblib.dump(svm, model_path)  # Save model
    return svm

# 修改 train_naive_bayes 函数
def train_naive_bayes(X_train, y_train, var_smoothing=1e-8):
    model_name = f"naive_bayes_model_vs{var_smoothing:.1e}.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(model_path):
        print(f"Model {model_name} already exists. Loading...")
        return joblib.load(model_path)

    nb = GaussianNB(var_smoothing=var_smoothing)
    print(f"Training Naive Bayes with var_smoothing={var_smoothing}...")
    nb.fit(X_train, y_train)

    joblib.dump(nb, model_path)  # Save model
    return nb