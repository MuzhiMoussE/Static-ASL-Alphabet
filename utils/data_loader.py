import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.feature_PCA import apply_pca
from utils.feature_hog import apply_hog
from tqdm import tqdm

DATASET_PATH = "data/asl_alphabet_train"
IMAGE_SIZE = (64, 64)
FEATURES_FILE = "data/features.npz"
FEATURES_FILE_TEST = "data/features_test.npz"
RESULTS_DIR = "results"

def load_asl_images(dataset_path, image_size=(64, 64), max_images_per_class=1000):
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))
    total_images = sum(len(os.listdir(os.path.join(dataset_path, label))[:max_images_per_class])
                       for label in classes if os.path.isdir(os.path.join(dataset_path, label)))

    with tqdm(total=total_images, desc="Loading images") as pbar:
        for label in classes:
            class_path = os.path.join(dataset_path, label)
            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)[:max_images_per_class]
            for img_file in images:
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                features, _ = apply_hog(gray)  # 只提取特征向量
                X.append(features)  # 确保每个特征向量是固定长度
                y.append(label)
                pbar.update(1)

    return np.array(X), np.array(y)

def preprocess_data(dataset_path=DATASET_PATH, image_size=IMAGE_SIZE, max_images_per_class=1000, n_components=100):
    if os.path.exists(FEATURES_FILE):
        print("Loading preprocessed features...")
        data = np.load(FEATURES_FILE)
        X, y = data["X"], data["y"]
    else:
        print("Extracting features...")
        X, y = load_asl_images(dataset_path, image_size, max_images_per_class)
        np.savez(FEATURES_FILE, X=X, y=y)
        print(f"Features saved to {FEATURES_FILE}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Splitting dataset...")
    # First split: Train (60%) and Temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    # Second split: Dev (20%) and Test (20%) from Temp
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )


    print("Applying PCA...")
    X_train_pca, X_dev_pca, X_test_pca, pca_model = apply_pca(X_train, X_dev, X_test, n_components=n_components)

    return X_train_pca, X_dev_pca, X_test_pca, y_train, y_dev, y_test, label_encoder, pca_model

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

    # Save HOG visualization
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    return features