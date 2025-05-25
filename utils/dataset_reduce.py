import os
from tqdm import tqdm

def reduce_dataset_by_interval(dataset_path, target_count=1000):
    """
    Reduce the dataset by keeping images at regular intervals to reach the target count.

    Args:
        dataset_path (str): Path to the dataset.
        target_count (int): Number of images to retain per class.
    """
    classes = sorted(os.listdir(dataset_path))
    for label in tqdm(classes, desc="Processing classes"):
        class_path = os.path.join(dataset_path, label)
        if not os.path.isdir(class_path):
            continue

        images = sorted(os.listdir(class_path))
        total_images = len(images)

        if total_images > target_count:
            # Calculate the interval for deletion
            interval = total_images // target_count
            retained_images = images[::interval][:target_count]

            # Delete images not in the retained list
            for img_file in images:
                if img_file not in retained_images:
                    os.remove(os.path.join(class_path, img_file))

    print(f"Dataset reduced. Each class now contains {target_count} images.")