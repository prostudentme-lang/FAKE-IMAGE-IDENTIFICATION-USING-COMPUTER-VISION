import cv2
import os

# Dataset folder
DATASET_PATH = "dataset"

# Blur threshold (adjust if needed)
BLUR_THRESHOLD = 100

removed = 0
total = 0

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD


for label in ["real", "fake"]:

    folder = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(folder):

        path = os.path.join(folder, img_name)

        total += 1

        try:
            img = cv2.imread(path)

            if img is None:
                os.remove(path)
                removed += 1
                continue

            h, w = img.shape[:2]

            # Remove very small images
            if h < 100 or w < 100:
                os.remove(path)
                removed += 1
                continue

            # Remove blurry images
            if is_blurry(img):
                os.remove(path)
                removed += 1
                continue

        except:
            os.remove(path)
            removed += 1

print("Total images checked:", total)
print("Removed bad images:", removed)
print("Remaining images:", total - removed)
