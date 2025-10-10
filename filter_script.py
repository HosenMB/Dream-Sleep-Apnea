import os
import cv2  # For image reading
import numpy as np
from skimage import filters, measure
from scipy.stats import skew, kurtosis
import shutil

# Function to compute Signal-to-Noise Ratio (SNR)
def compute_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    return signal / noise if noise != 0 else 0

# Function to compute entropy
def compute_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 1), density=True)
    histogram = histogram + 1e-10  # avoid log(0)
    return -np.sum(histogram * np.log2(histogram))

# Function to compute contrast
def compute_contrast(image):
    return np.std(image) / np.mean(image)

# Function to calculate key image metrics
def calculate_image_metrics(image):
    image = image.astype(float) / 255.0  # Normalize to [0,1] range
    snr = compute_snr(image)
    entropy = compute_entropy(image)
    contsaverast = compute_contrast(image)
    img_skew = skew(image.flatten())
    img_kurtosis = kurtosis(image.flatten())
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    return snr, entropy, contrast, img_skew, img_kurtosis, min_intensity, max_intensity

# Filtering function based on dynamic threshold
def check_criteria(criteria, threshold):
    return sum(criteria) >= threshold

# Filtering function for Apnea images (threshold increased by 1)
def filter_apnea_image(image_metrics):
    snr, entropy, contrast, img_skew, img_kurtosis, min_intensity, max_intensity = image_metrics
    # Check conditions for Apnea images
    criteria = [
        snr < 7.5,
        entropy > 5.9,
        contrast > 0.55,
        img_skew > 0.05,
        img_kurtosis > 0.2,
        min_intensity < 0.23,
        max_intensity > 0.75
    ]
    # Apply filter with increased threshold (5 instead of 4)
    return check_criteria(criteria, threshold=3)

# Filtering function for Non-Apnea images (threshold increased by 2)
def filter_non_apnea_image(image_metrics):
    snr, entropy, contrast, img_skew, img_kurtosis, min_intensity, max_intensity = image_metrics
    # Check conditions for Non-Apnea images
    criteria = [
        snr > 8.5,
        entropy < 5.9,
        contrast < 0.5,
        img_skew < 0.05,
        img_kurtosis < 0,
        min_intensity > 0.26,
        max_intensity < 0.75
    ]
    # Apply filter with increased threshold (6 instead of 4)
    return check_criteria(criteria, threshold=6)

# Main function to process images and apply filtering
def filter_images(input_folder, output_folder, is_apnea):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        if img is None:
            continue

        # Calculate image metrics
        metrics = calculate_image_metrics(img)

        # Apply filters based on whether it's an Apnea or Non-Apnea folder
        if is_apnea and filter_apnea_image(metrics):
            shutil.copy(img_path, os.path.join(output_folder, img_file))
        elif not is_apnea and filter_non_apnea_image(metrics):
            shutil.copy(img_path, os.path.join(output_folder, img_file))


# Driver function to run the script
if __name__ == "__main__":
    # Set the input folders for Apnea and Non-Apnea images
    apnea_input_folder = "./Binary_Classification_Apnea/apnea"
    non_apnea_input_folder = "./Binary_Classification_Apnea/non_apnea"

    # Set the output folders for filtered Apnea and Non-Apnea images
    apnea_output_folder = "./BCA/apnea"
    non_apnea_output_folder = "./BCA/non_apnea"

    # Filter Apnea images
    print("Filtering Apnea images...")
    filter_images(apnea_input_folder, apnea_output_folder, is_apnea=True)

    # Filter Non-Apnea images
    print("Filtering Non-Apnea images...")
    filter_images(non_apnea_input_folder, non_apnea_output_folder, is_apnea=False)

    print("Filtering complete!")

