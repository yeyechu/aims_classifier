import os
from PIL import Image
from collections import defaultdict


def analyze_dataset_train(directory):
    class_counts = defaultdict(int)
    widths = []
    heights = []
    total_images = 0

    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path) and class_name.startswith("no"):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        widths.append(img.width)
                        heights.append(img.height)
                        class_counts[class_name] += 1
                        total_images += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return class_counts, widths, heights, total_images


def analyze_dataset_test(directory):
    widths = []
    heights = []
    total_images = 0

    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                total_images += 1
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    return widths, heights, total_images