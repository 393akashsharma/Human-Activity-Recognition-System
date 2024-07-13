# Human Action Recognition

## Project Overview
This project aims to develop a human action recognition system using machine learning techniques. The system classifies actions from images into predefined categories. The project explores various feature extraction methods and machine learning models to achieve high accuracy.

## Directory Structure
Human Action Recognition/
│
├── train/ # Directory containing training images
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Training_set.csv # CSV file containing filenames and labels
├── models/ # Directory to save trained models
├── results/ # Directory to save results
├── README.md # This README file
└── main.py # Main script to run the project


## Dependencies
The project requires the following libraries:
- pandas
- numpy
- scikit-learn
- OpenCV
- scikit-image
- keras
- tensorflow

Install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn opencv-python-headless scikit-image keras tensorflow

Data Preparation
Training Images: Place all the training images in the train directory.
Training Set CSV: The Training_set.csv file should have the following format:
csv
filename,label
image1.jpg,action1
image2.jpg,action2
...
Feature Extraction Methods
The project explores various feature extraction methods:

Pixel Intensity: Directly using pixel values of resized grayscale images.
SIFT (Scale-Invariant Feature Transform): Extracts key points and descriptors from images.
HOG (Histogram of Oriented Gradients): Captures edge or gradient structures.
LBP (Local Binary Patterns): Captures texture features.
Deep Features: Using a pre-trained VGG16 model to extract high-level features from images.
Methodologies
The project implements the following methodologies:

Pixel Intensity
Load and resize images to a common size.
Flatten the image to create feature vectors.
Normalize features using StandardScaler.
Encode labels using LabelEncoder.
Apply PCA to reduce dimensionality.
Train an SVM model with an RBF kernel.
Evaluate the model on a validation set.
SIFT + HOG
Compute SIFT and HOG features for each image.
Concatenate the features to create combined feature vectors.
Normalize features using StandardScaler.
Encode labels using LabelEncoder.
Apply PCA to reduce dimensionality.
Train an SVM model with an RBF kernel.
Evaluate the model on a validation set.
SIFT + HOG + LBP
Compute SIFT, HOG, and LBP features for each image.
Concatenate the features to create combined feature vectors.
Normalize features using StandardScaler.
Encode labels using LabelEncoder.
Apply PCA to reduce dimensionality.
Train an SVM model with an RBF kernel.
Evaluate the model on a validation set.
Deep Features
Use a pre-trained VGG16 model to extract deep features from images.
Normalize features using StandardScaler.
Encode labels using LabelEncoder.
Train an SVM model with an RBF kernel.
Evaluate the model on a validation set.
Usage
Clone the repository.
Prepare the data as described above.
Run the main.py script to train and evaluate the model.
bash
Copy code
python main.py
Evaluation
The model's performance is evaluated using accuracy and classification reports. The results will be printed on the console.

Accuracy: 0.85
Classification Report:
              precision    recall  f1-score   support

     action1       0.84      0.88      0.86       123
     action2       0.87      0.81      0.84       107
        ...       ...       ...       ...       ...

   micro avg       0.85      0.85      0.85       230
   macro avg       0.85      0.85      0.85       230
weighted avg       0.85      0.85      0.85       230
Results
The results, including the trained model and evaluation metrics, will be saved in the results and models directories.

Acknowledgements
The project utilizes various machine learning and image processing libraries, including OpenCV, scikit-image, scikit-learn, keras, and tensorflow.

This README file provides a detailed overview of the project, ensuring that users can easily understand the purpose, setup, and usage of the human action recognition system.

