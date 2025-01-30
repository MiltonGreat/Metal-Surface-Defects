# Metal-Surface-Defects

### Overview

This project uses machine learning techniques to predict the severity of defects in manufacturing. The goal is to classify images of metal surfaces into different defect categories based on the type of defect observed. A Convolutional Neural Network (CNN) model is employed for image classification, where the input consists of images of metal surfaces with six possible defect types.

### Objectives

The primary objectives of this project are:

- Defect Classification: Classify images of metal surfaces into different defect categories.
- Severity Prediction: Predict the severity of defects using machine learning techniques.
- Model Evaluation: Evaluate the performance of the CNN model and analyze its predictions using various metrics.

### Dataset

The dataset consists of images from the NEU Metal Surface Defects Data set, which contains 1,800 images classified into six defect types. The images are in BMP format and have been resized to a consistent size of 224x224 pixels.

- Training Data: 1,324 images
- Test Data: 332 images
    
The images are categorized into several defect types such as:

- Crazing
- Inclusion
- Patches
- Pitted
- Rolled
- Scratches

The dataset is structured into train, test, and validation directories.

### Project Workflow

1. Data Loading and Preprocessing:
- Images are loaded, resized to 224x224 pixels, and normalized to the range [0, 1].
- The labels (defect types) are encoded using label encoding to transform categorical data into numeric form.
- The dataset is split into training and test sets (80% training, 20% testing).

2. Model Architecture:
- A Convolutional Neural Network (CNN) is built with two convolutional layers followed by max-pooling, flattening, and fully connected layers.
- The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss, suitable for multi-class classification.

3. Model Training:
- The model is trained for 10 epochs with a batch size of 32, and validation is performed on the test set.

4. Model Evaluation:
- The model's accuracy is evaluated on the test set.
- The classification report and confusion matrix provide insights into the precision, recall, and F1-score of each defect type.
    
### Results

![screenshot-localhost_8888-2025 01 30-10_10_32](https://github.com/user-attachments/assets/06d45cd3-4bd0-4df2-942f-618b8682221b)

The model achieved a test accuracy of 89.16%, with performance varying across different defect types. The classification report provides detailed metrics for each defect type, including precision, recall, and F1-score.

### Future Work

- Explore more advanced models: Experiment with more complex models like Deep Neural Networks or pre-trained CNNs (e.g., VGG16 or ResNet) for potentially better performance.
- Data augmentation: Use techniques like flipping, rotating, and scaling to further augment the training dataset.

### Source

Dataset: [Metal Surface Defects Dataset on Kaggle](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)

https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data
