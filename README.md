# Metal-Surface-Defects

### Overview

In the manufacturing industry, identifying and predicting defects early in the production process is essential to improve product quality, reduce costs, and enhance overall efficiency. This project utilizes machine learning techniques to predict the severity of defects in manufacturing based on data from various factors such as:

- Defect Type: The type of defect observed in the product (e.g., scratches, cracks).
- Defect Location: The part of the product or surface where the defect occurs (e.g., internal or surface defects).
- Inspection Method: The method used for inspecting the product (e.g., visual inspection, automated testing).
- Repair Cost: The estimated cost required for repair.

The data used in this project includes simulated images of surface defects from a manufacturing setting, with the aim to train a classification model that can predict the severity of defects.

### Dataset

The dataset used in this project is the Metal Surface Defects dataset, available in the form of BMP images representing different types of defects. The images are categorized into several defect types such as:

- Crazing
- Inclusion
- Patches
- Pitted
- Rolled
- Scratches

The dataset is structured into train, test, and validation directories.

### Example Image Categories

- train/Crazing
- train/Inclusion
- train/Patches
- train/Pitted
- train/Rolled
- train/Scratches

The project involves image classification to detect and classify these defects based on their characteristics.

### Key Steps

Data Loading and Preprocessing:
- The images are loaded from the train, test, and validation directories.
- Preprocessing steps like resizing and flattening are applied to prepare the image data for machine learning models.
- Label encoding is used to handle categorical features such as defect type, location, and severity.

Model Building:
- Two types of models were explored: Dense models (using flattened images) and Convolutional Neural Networks (CNNs).
- For dense models, images are flattened into vectors before being fed into the model.
- CNN models take advantage of spatial relationships in the images to perform better.

Model Evaluation:
- The models are evaluated using various metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- Hyperparameter tuning is performed using GridSearchCV to optimize model performance.

SMOTE:
- SMOTE is used to handle class imbalance by generating synthetic samples for the minority class.

### Results

The model was able to achieve a classification accuracy of 74.4%, with a good balance of precision and recall across different defect types. The confusion matrix and classification report provide insights into model performance for each defect type.

### Hyperparameter Tuning

A Random Forest classifier was used, and the model's hyperparameters were tuned using GridSearchCV to improve the performance. The tuned model provided an accuracy improvement and better generalization to unseen data.

### Future Work

- Explore more advanced models: Experiment with more complex models like Deep Neural Networks or pre-trained CNNs (e.g., VGG16 or ResNet) for potentially better performance.
- Data augmentation: Use techniques like flipping, rotating, and scaling to further augment the training dataset.
- Deployment: Deploy the model in a web application or integrate it into the manufacturing pipeline to automate defect severity prediction.

### Source

https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data
