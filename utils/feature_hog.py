from skimage.feature import hog
#from plot_utils import visualize_hog
def apply_hog(gray):
    # Extract HOG features
    features, hog_image = hog(
        gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        orientations=9, block_norm='L2-Hys', feature_vector=True, visualize=True
    )
    #visualize_hog(gray)  # Save the HOG visualization
    return features, hog_image