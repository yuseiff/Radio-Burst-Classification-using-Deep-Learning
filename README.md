# Radio Burst Classification using Deep Learning

## Overview

This project focuses on the classification of radio burst spectrogram images using various deep learning techniques. It was developed during a 6-month AI research internship at the Egyptian Space Agency (EGSA) under the supervision of Eng. Hassan Noureldeen and Dr. Mohamed Nedal. The primary goal was to create a robust system capable of automatically classifying radio bursts into different types, demonstrating both high-speed and accurate detection capabilities.

The work encompasses a range of experiments and implementations including:

1.  Training a custom Convolutional Neural Network (CNN).
2.  Experimenting with pre-trained deep learning models via Transfer Learning
3.  Utilizing state-of-the-art Vision Transformers (ViT, Swin, and ConvNeXt)
4.  Employing YOLO models (v10 and v11 variants) for real-time object detection.
5.  Developing an automated system for continuous data capture and classification.

## Project Structure

This repository contains the following main components:

*   **`webscraping.py`**: A Python script for web scraping and file downloading from specified URLs.
*   **`read_Data.py`**: Python functions that handle reading different data types (PNG, FITS) and processing them, including resizing and color space conversion.
*   **`Yolo.ipynb`**: A Jupyter notebook encompassing the implementation of radio burst classification using YOLOv10 and YOLOv11 models. It covers model setup, training, evaluation, and visualization.
*   **`app.py`**: A Python script designed to run continuously, monitoring a directory for new radio burst images, classifying them using a YOLO model, and saving them into designated directories based on the outputted class.
*   **`Documentation for prototype.ipynb`**: A Jupyter Notebook that implement a deep learning pipeline for classifying spectrogram images of radio bursts into three types and includes a lot of preprocessing and hyperparameter tuning.
*   **`png prototype.ipynb`**: A Jupyter notebook implements image annotations and organizes them into a format suitable for training YOLO object detection models. It performs key tasks like converting annotations to YOLO format, organizing datasets, and generating config files.
*   **`Documentation: YOLO Annotation and Dataset Preparation Script`**: explains the `png prototype.ipynb` script and the YOLO annotation format and data preprocessing steps.
*   **`Project.ipynb`**: A Python notebook encompassing several tasks, including custom CNN model training, evaluation, and visualization, as well as K-Fold cross-validation on several pre-trained models.
*   **`read_data.py`**: A Python script including functions for reading both PNG image files and FITS files.
*   **`Data`**: Contains dataset files and a config file (`data.yaml`).
*   **`Weight`**: Contains the trained weights of the models.
*   **`Classified`**: a folder that contains labeled subfolders for saving classified radio burst images by type.
*   **`Test`**: a folder for the images that will be used to test the classification models.
*   **`cv_results.csv`**: CSV file containing the cross validation results for each pretrained model.
*   **`final_results.csv`**: CSV file containing the final results of each classification method.
*  **`tfhub_cache`**: folder to cache TensorFlow Hub models

## Data Collection and Preparation

The initial dataset was acquired from the e-Callisto network using the `webscraping.py` script, which is designed to automatically download specific files from the specified URL. The extracted data included both PNG and FITS files and was then processed and formatted for use in the different experiments and models.

Key processes include:

*   Reading PNG image files, converting them to RGB, and resizing them using OpenCV (implemented in `read_Data.py`).
*   Reading FITS files, extracting the data matrix, frequency axis, and time axis using `astropy.io.fits` (implemented in `read_Data.py`).
*   Plotting FITS data using `matplotlib.pyplot` both before and after background filtering.
*   Preprocessing of spectrogram data by subtracting median values for noise reduction.
*   Splitting data into training, validation, and test sets, and encoding class labels to numerical IDs (implemented in `Documentation: YOLO Annotation and Dataset Preparation Script`).
*   Generating a YAML configuration file (`data.yaml`) for YOLO training.

## Deep Learning Implementation

### 1. Custom CNN Model
  
A Convolutional Neural Network was defined with three convolutional layers with ReLU activation, followed by max-pooling and dropout. A fully connected dense layer, followed by dropout, and an output layer with a softmax activation for classification were also implemented.
The model was trained on the preprocessed data using the Adam optimizer and sparse categorical cross-entropy loss, and checkpointing was used to save the best-performing model during training.

### 2. Pre-trained Models

This implementation utilized several pre-trained models from the TensorFlow's Keras applications to leverage transfer learning. The following models were used:
* VGG16
* DenseNet121
* MobileNet
* MobileNetV2

Each pre-trained model's base layers were frozen and a custom classification head (dense layers followed by a softmax output) was added to perform the classification for our three classes. The K-Fold cross-validation (5 folds) was used to evaluate each pretrained model and the results were saved to the `cv_results.csv` file.
### 3. Vision Transformers (ViT, Swin, and ConvNeXt)

This project uses Vision Transformers (ViT), Swin Transformer, and ConvNext to investigate the performance of state-of-the-art models on classifying spectrogram data. These models, implemented using TensorFlow Hub, each include:
* A resizing layer
* A feature extractor (vit, swin, or convnext)
* A dense classification head

Each model was trained using an early stopping mechanism to avoid overfitting and the weights were saved for further use. Evaluation metrics such as train and test accuracies are reported in the code itself.
### 4. YOLO Models (v10 and v11 Variants)

YOLO (You Only Look Once) models, particularly the v10 and v11 variants, were employed to automate the process of object detection and classification of radio burst images and the following configurations were used:
*   YOLOv10 Nano was trained using the specifications found in the documentation provided in the repository.
*   YOLOv11 Nano was trained using the specifications found in the documentation provided in the repository.
*   YOLOv11 Small was trained using the specifications found in the documentation provided in the repository.
*   YOLOv11 large was trained using the specifications found in the documentation provided in the repository.
*   YOLOv11 xlarge was trained using the specifications found in the documentation provided in the repository.

  A separate Python script (`app.py`) runs continuously, monitors a specific directory, and classifies the incoming images. The trained model weights are saved for future use.
  
## Code Documentation

The repository includes detailed code documentation, explaining each step and process and the following steps are implemented in detail in the `Project.ipynb`, `Documentation for prototype.ipynb`, and `Documentation: YOLO Annotation and Dataset Preparation Script` files:

*   **Preprocessing**: Includes the methods used to manipulate the raw data so that they are suitable for model training.
*   **Model Training**: Covers the steps involved in training each model including the selection of hyperparameters.
*   **Evaluation**: Details the evaluation processes used to assess the model's performance such as precision, recall, and F1 score, as well as confusion matrix visualisation.
*  **Visualization**: Showcases techniques to visualize the input data and the classification results.

## Key Outputs
The project yielded several important outputs:

-   **Trained Model Weights**: Saved for each trained model, both custom CNN and pre-trained, and also for the YOLO models in a specified directory for future use.
-   **Performance Metrics**: Outputs from cross-validation for the pre-trained models, and both training and testing metrics for CNN, ViT, Swin, ConvNext, and the Ensemble method are printed in the code.
-   **Final Results**: CSV file containing the final performance results (`final_results.csv`).
-   **Cross-Validation Results**:  CSV file containing the cross-validation performance results for the pre-trained models (`cv_results.csv`).
 -  **Visual Results**: Images with detected objects and classifications saved by the YOLO model.

## Key Concepts
The core concepts implemented in this project include:

-   **YOLO Annotation Format**:  YOLO annotations use a simple text format where each line corresponds to an object in the image. The format is: `<class_id> <center_x> <center_y> <width> <height>`.
  All values (except class_id) are normalized to the range [0, 1] with respect to the
image dimensions.
-   **Data Preprocessing**:  Includes resizing, color space conversion, and noise removal through median subtraction.
-   **Error Handling**: The script logs errors (e.g., missing images or annotations) to facilitate debugging.
-   **Data Organization**: The scripts ensure that the data is properly formatted and split into training and validation sets.
-   **Multi-threading**:  The scripts implement multi-threading in some functionalities to optimize data loading and preprocessing.

## Conclusion

This project demonstrates the application of deep learning techniques, including custom CNNs, pretrained models, and state-of-the-art architectures like Vision Transformers, for classifying radio burst spectrograms. Results from various methods
are compared to identify the best-performing model.