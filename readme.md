# Apnea Detection Dataset Preparation

This repository focuses on preparing a high-quality dataset for apnea detection, sourced from the [Physionet apnea dataset]((https://physionet.org/content/apnea-ecg/1.0.0/)). It encompasses various preprocessing and filtering steps designed to enhance the quality of the images, ensuring optimal performance in deep learning model training.

By systematically processing the dataset, we aim to improve the accuracy and reliability of apnea detection models, facilitating advancements in sleep health diagnostics.

## 1. Dataset Generation

In the **Physionet_Apnea_Dataset_preparation notebook**, the Continuous Wavelet Transform (CWT) spectrogram dataset was generated from the Physionet apnea dataset. This process involved extracting relevant features and creating spectrogram images for both apnea and non-apnea events.

### Sample Image
![Sample CWT Spectrogram](./figures/average_spectogram.png)

## 2. Exploratory Data Analysis (EDA)

In the **EDA notebook**, a detailed analysis was conducted on the generated dataset. Key findings led to the identification of filtering criteria to enhance the quality of apnea and non-apnea images. The following metrics were considered during the analysis:

- **Signal-to-Noise Ratio (SNR)**
- **Entropy**
- **Contrast**
- **Skewness**
- **Kurtosis**
- **Intensity Ranges**

## 3. Image Filtering

Based on the findings from the EDA, specific filtering criteria were applied to both apnea and non-apnea images. This step was performed in the **Image Filtering notebook**, where images were processed to retain only those meeting the specified quality thresholds.

### Filtering Criteria:
- **Apnea Images**: Selected based on SNR, entropy, contrast, skewness, kurtosis, and intensity ranges.
- **Non-Apnea Images**: Selected based on different criteria focusing on maintaining high-quality images.

## 4. Final Dataset

The final filtered dataset is now ready for use in deep learning models, providing improved training and evaluation data for apnea detection tasks.

### Usage
To use this dataset in your deep learning model, please refer to the specific model implementation files included in this repository.