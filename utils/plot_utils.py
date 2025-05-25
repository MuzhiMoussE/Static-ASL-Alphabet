
from skimage.feature import hog
from skimage import exposure
from evaluate_models import evaluate_model
import os
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from models import train_knn, train_svm, train_naive_bayes
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from models import train_knn, train_svm, train_naive_bayes
def plot_pca_2d(X, y, label_names, title="PCA Projection"):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 8))

    unique_classes = np.unique(y)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))

    for i, label in enumerate(unique_classes):
        class_points = X_pca[y == label]
        plt.scatter(class_points[:, 0], class_points[:, 1], s=20, color=colors[i], label=label_names[label])

        # Add center point
        center = class_points.mean(axis=0)
        plt.scatter(center[0], center[1], s=100, color=colors[i], edgecolor='black', marker='X', label=f"{label_names[label]} Center")

        # Add ellipse boundary
        cov = np.cov(class_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor=colors[i], facecolor='none', linestyle='--', linewidth=1.5)
        plt.gca().add_patch(ellipse)

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/pca_projection.png")
    plt.show()

def visualize_hog(image):
    # Compute HOG features and the visualization image
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Create a subplot to display the original image and HOG visualization
    plt.figure(figsize=(12, 6))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Display the HOG visualization
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.axis("off")

    # Save and show the combined visualization
    plt.tight_layout()
    plt.savefig(os.path.join("results", "hog_features.png"))
    plt.show()



def plot_knn_performance(X_dev,X_test, y_dev,y_test, k_values, results_dir="results"):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    accuracies = []
    for k in k_values:

        # Train KNN using the existing train_knn function
        knn = train_knn(X_dev, y_dev, n_neighbors=k)
        # Evaluate the model using the existing evaluate_model function
        metrics = evaluate_model(knn, X_test, y_test, name=f"KNN (k={k})")
        acc = metrics['accuracy']  # Extract accuracy from the metrics dictionary
        accuracies.append(acc)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='g')
    plt.title("KNN Performance with Different n_neighbors Values")
    plt.xlabel("n_neighbors (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "knn_performance.png"))
    plt.show()

def plot_svm_performance(X_dev,X_test, y_dev,y_test, c_values, results_dir="results"):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    accuracies = []
    for c in c_values:
        # Train SVM using the existing train_svm function
        svm = train_svm(X_dev, y_dev, C=c, kernel='rbf')

        # Evaluate the model using the existing evaluate_model function
        metrics = evaluate_model(svm, X_test, y_test, name=f"SVM (C={c})")
        acc = metrics['accuracy']  # Extract accuracy from the metrics dictionary
        accuracies.append(acc)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(c_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title("SVM Performance with Different C Values")
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Accuracy")
    plt.xscale("log")  # Use logarithmic scale for C values
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "svm_performance.png"))
    plt.show()

# 测试不同 var_smoothing 的性能
def plot_naive_bayes_performance(X_dev, X_test, y_dev, y_test, var_smoothing_values, results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    accuracies = []
    for var_smoothing in var_smoothing_values:
        # 训练 Naive Bayes 模型
        nb = train_naive_bayes(X_dev, y_dev, var_smoothing=var_smoothing)

        # 在测试集上评估模型
        metrics = evaluate_model(nb, X_test, y_test, name=f"Naive Bayes (var_smoothing={var_smoothing:.1e})")
        acc = metrics['accuracy']
        accuracies.append(acc)

    # 绘制结果
    plt.figure(figsize=(8, 6))
    plt.plot(var_smoothing_values, accuracies, marker='o', linestyle='-', color='r')
    plt.title("Naive Bayes Performance with Different var_smoothing Values")
    plt.xlabel("var_smoothing")
    plt.ylabel("Accuracy")
    plt.xscale("log")  # 使用对数刻度
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "naive_bayes_performance.png"))
    plt.show()





def plot_learning_curve(model_name, X_train, y_train, X_test, y_test, train_sizes=None, results_dir="results"):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)  # 默认从 10% 到 100% 的训练数据

    train_sizes = (train_sizes * len(X_train)).astype(int)  # 转换为绝对大小
    train_accuracies = []
    test_accuracies = []

    # 随机打乱训练数据
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    for size in train_sizes:
        # 划分训练子集
        X_subset = X_train[:size]
        y_subset = y_train[:size]

        # 训练模型
        if model_name == "knn":
            model = train_knn(X_subset, y_subset)
        elif model_name == "svm":
            model = train_svm(X_subset, y_subset)
        elif model_name == "naive_bayes":
            model = train_naive_bayes(X_subset, y_subset)
        else:
            raise ValueError(f"Error: {model_name}")

        # 在训练集和测试集上评估
        train_pred = model.predict(X_subset)
        test_pred = model.predict(X_test)

        train_accuracies.append(accuracy_score(y_subset, train_pred))
        test_accuracies.append(accuracy_score(y_test, test_pred))

    # 绘制学习曲线
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_accuracies, label="accuracy of training", marker="o")
    plt.plot(train_sizes, test_accuracies, label="accuracy of testing", marker="s")
    plt.title(f"Learning Curve ({model_name.upper()})")
    plt.xlabel("Size of Training Set")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 保存图像
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_path = os.path.join(results_dir, f"learning_curve_{model_name}.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Learning curve has been saved to {plot_path}")


def plot_confusion(models, X_test, y_test, label_encoder, results_dir="results"):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for model_name in models:
        # Train or load the model
        if model_name == "knn":
            model = train_knn(X_test, y_test, n_neighbors=5)
        elif model_name == "svm":
            model = train_svm(X_test, y_test, C=1.0, kernel='rbf')
        elif model_name == "naive_bayes":
            model = train_naive_bayes(X_test, y_test, var_smoothing=1e-8)
        else:
            print(f"Unsupported model: {model_name}")
            continue

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Generate and save the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title(f"{model_name.upper()} Confusion Matrix")
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"conf_matrix_{model_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion matrix for {model_name} saved to {plot_path}")
