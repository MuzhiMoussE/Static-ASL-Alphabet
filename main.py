import argparse
import os
import joblib
from skimage.feature import hog
from utils.plot_utils import plot_learning_curve

from models import train_knn, train_svm, train_naive_bayes
from evaluate_models import evaluate_model
from utils.data_loader import preprocess_data
from utils.feature_hog import apply_hog
from utils.plot_utils import plot_svm_performance
from utils.plot_utils import plot_knn_performance
from utils.plot_utils import plot_naive_bayes_performance
import argparse
import os
import cv2
import numpy as np
from utils.data_loader import preprocess_data
from utils.plot_utils import visualize_hog
from utils.nested_CV import run_nested_cv_all_models
from utils.plot_utils import plot_confusion
MODELS_DIR = "models"

def load_model(model_name):
    """Load a saved model from the models directory."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        return joblib.load(model_path)
    else:
        print(f"Model {model_name} not found. Please train the model first.")
        return None

def compare_models(X_train, X_test, y_train, y_test, n_neighbors=5, c_svm=1.0):
    results = {}

    # Train & Evaluate KNN
    knn = train_knn(X_train, y_train, n_neighbors)
    results["KNN"] = evaluate_model(knn, X_test, y_test, "KNN")

    # Train & Evaluate SVM
    svm = train_svm(X_train, y_train, C=c_svm, kernel='rbf')
    results["SVM"] = evaluate_model(svm, X_test, y_test, "SVM")

    # Train & Evaluate Naive Bayes
    nb = train_naive_bayes(X_train, y_train)
    results["Naive Bayes"] = evaluate_model(nb, X_test, y_test, "Naive Bayes")
    return results

def preprocess_image(image_path, image_size=(64, 64)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.resize(img, image_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features, hog_image = apply_hog(gray)
    return features

def predict_image(model, image_path, label_encoder, pca_model):
    # Preprocess the image to extract HOG features
    features = preprocess_image(image_path)
    features = features.reshape(1, -1)  # Reshape for prediction

    # Apply PCA transformation
    features_pca = pca_model.transform(features)

    # Predict the class
    prediction = model.predict(features_pca)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

def main():
    parser = argparse.ArgumentParser(description="Train or use existing models for ASL classification.")
    parser.add_argument("--use_model", type=str, choices=["knn", "svm", "naive_bayes"],
                        help="Specify the model to use (knn, svm, naive_bayes). If not provided, a new model will be trained.")
    parser.add_argument("--n_neighbor", type=int, default=5, help="Number of neighbors for KNN (default: 5).")
    parser.add_argument("--c_svm", type=float, default=1.0, help="Regularization parameter C for SVM (default: 1.0).")
    parser.add_argument("--plot_svm", action="store_true", help="Generate and save the SVM performance plot.")
    parser.add_argument("--plot_knn", action="store_true", help="Generate and save the KNN performance plot.")
    parser.add_argument("--plot_naive_bayes", action="store_true", help="Generate and save the Naive Bayes performance plot.")
    parser.add_argument("--predict", type=str, help="Path to the image for prediction.")
    parser.add_argument("--run_cv", action="store_true", help="Run nested cross-validation for all models.")
    parser.add_argument("--learning_curve", action="store_true", help="Plot learning curves for the models.")
    parser.add_argument("--plot_confusion", action="store_true", help="Generate and save the confusion matrix.")

    args = parser.parse_args()

    # Load and preprocess data
    X_train,X_dev, X_test, y_train,y_dev, y_test, label_encoder, pca_model = preprocess_data()

    if args.plot_svm:
        plot_svm(X_train, X_dev, y_train, y_dev)
    if args.plot_knn:
        plot_knn(X_train, X_dev, y_train, y_dev)
        #print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        #print(f"X_dev shape: {X_dev.shape}, y_dev shape: {y_dev.shape}")
        #print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    if args.plot_naive_bayes:
        plot_naive_bayes(X_dev, X_test, y_dev, y_test)
    if args.run_cv:
        # 合并 train 和 dev 用于 CV（test 仍然留着不碰）
        X_cv = np.vstack((X_train, X_dev))
        y_cv = np.hstack((y_train, y_dev))
        run_nested_cv_all_models(X_cv, y_cv)
        return  # 运行完 CV 后退出
    if args.learning_curve:
        plot_learning_curve("knn", X_train, y_train, X_test, y_test)
        plot_learning_curve("svm", X_train, y_train, X_test, y_test)
        plot_learning_curve("naive_bayes", X_train, y_train, X_test, y_test)
    if args.plot_confusion:
        models = ["knn", "svm", "naive_bayes"]
        plot_confusion(models, X_test, y_test, label_encoder)

    elif args.use_model:
        # Load the specified model
        if args.use_model == "knn" :
            model_name = f"knn_model_n1.pkl"
        elif args.use_model == "svm":
            model_name = f"svm_model_C10_kernelrbf.pkl"
        elif args.use_model == "naive_bayes":
            model_name = f"naive_bayes_model_vs1.0e-08.pkl"
        else: model_name = f"{args.use_model}_model.pkl"
        model = load_model(model_name)
        if model:

            # Load test data from file
            test_data_file = "data/test_data.npz"
            if os.path.exists(test_data_file):
                test_data = np.load(test_data_file)
                X_test, y_test = test_data["X_test"], test_data["y_test"]
            if args.predict:

                # Load label encoder
                X_train, X_dev, X_test, y_train, y_dev, y_test, label_encoder, pca_model = preprocess_data()

                # Predict the uploaded image
                print(f"Predicting the class of the image: {args.predict}...")
                predicted_label = predict_image(model, args.predict, label_encoder,pca_model)
                print(f"Predicted Character: {predicted_label}")
            else:
                print("Please specify both --use_model and --predict arguments.")


    else:
        # Train and compare models
        print("No model specified. Training and comparing models...")
        compare_models(X_train, X_test, y_train, y_test, n_neighbors=args.n_neighbor, c_svm=args.c_svm)

def plot_svm(X_dev, X_test, y_dev, y_test):
    # Generate and save the SVM performance plot
    print("Generating SVM performance plot...")
    c_values = [0.01, 0.1, 1, 10, 100]
    plot_svm_performance(X_dev,X_test, y_dev,y_test, c_values)

def plot_knn(X_dev, X_test, y_dev, y_test):
    # Generate and save the KNN performance plot
    print("Generating KNN performance plot...")
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    plot_knn_performance(X_dev,X_test, y_dev,y_test, k_values)
def plot_naive_bayes(X_dev, X_test, y_dev, y_test):
    print("Generating Naive Bayes performance plot...")
    var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    plot_naive_bayes_performance(X_dev, X_test, y_dev, y_test, var_smoothing_values)

if __name__ == "__main__":
    main()