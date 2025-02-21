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

![screenshot-www kaggle com-2025 02 03-14_38_17](https://github.com/user-attachments/assets/d478455d-380b-4997-847e-948ee49491ad)
    
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
    
### Key Insights:

- **Class 0 (RS)** is well predicted, with very few misclassifications (1 misclassified as Class 2).
- **Class 1 (Pa)** has 8 instances misclassified as Class 3 and 2 as Class 5, indicating it’s sometimes confused with other defects.
- **Class 2 (Cr)** is almost perfectly predicted, with only 1 misclassification as Class 0 and 1 misclassified as Class 3.
- **Class 3 (PS)** is more challenging, with some misclassifications (2 as Class 1 and 5 as Class 5), which aligns with its lower precision and higher recall.
- **Class 4 (In)** is perfectly predicted, with no misclassifications.
- **Class 5 (Sc)** has some misclassifications (10 as Class 1, 6 as Class 3), which aligns with its lower recall.

### Results:

![screenshot-localhost_8888-2025 01 30-10_10_32](https://github.com/user-attachments/assets/06d45cd3-4bd0-4df2-942f-618b8682221b)

- Accuracy of 89% is quite good for a multi-class classification problem, but class imbalances (i.e., some classes are more frequent than others) and misclassifications in certain defect types (like Class 3 (PS) and Class 5 (Sc)) indicate areas for improvement.
- The F1-scores suggest that Class 0, 2, and 4 are well classified, while Class 1, 3, and 5 could benefit from further attention, especially in terms of precision and recall balance.

### Suggestions for Improvement:

- **Class Imbalance**: Consider techniques like class weighting or SMOTE to improve performance on minority classes (like Class 3 and 5).
- **Model Tuning**: Adjust model parameters (e.g., regularization) to reduce false positives/negatives for certain classes.
- **Data Augmentation**: Since this is an image classification problem, using data augmentation techniques (like rotating, zooming, or flipping the images) could help improve generalization, especially for underperforming classes.

### Source

Dataset: [Metal Surface Defects Dataset on Kaggle](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)
