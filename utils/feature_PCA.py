from sklearn.decomposition import PCA
#from plot_utils import plot_pca_2d
import numpy as np

def apply_pca(X_train,X_dev ,X_test, n_components=100):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_dev_pca = pca.transform(X_dev)
    X_test_pca = pca.transform(X_test)
    # 可视化 PCA 降维结果
    #if not os.path.exists(RESULTS_DIR):
    #    os.makedirs(RESULTS_DIR)
    #plot_pca_2d(np.vstack((X_train_pca, X_test_pca)), np.hstack((y_train, y_test)),
    #            label_encoder.classes_, title="PCA Projection")
    #print(f"PCA visualization saved in {RESULTS_DIR}")
    return X_train_pca, X_dev_pca,X_test_pca, pca