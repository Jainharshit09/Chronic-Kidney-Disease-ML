# Chronic-Kidney-Disease-ML

Chronic Kidney Disease Prediction using PCA, ANN (Keras), Random Forest, and Imbalanced Class Handling

This Colab Notebook explores the use of machine learning to predict chronic kidney disease (CKD), focusing on addressing class imbalance and improving model generalizability.

### **Data Preparation**

### **Data Loading**: Loads a dataset containing features relevant to CKD.

### **Cleaning and Preprocessing**: Performs necessary cleaning steps to ensure data quality, such as handling missing values and outliers.

### **Exploratory Data Analysis (EDA)**:
Analyzes the distribution of features to understand patterns and relationships.
Investigates class imbalance, where the distribution of healthy and CKD patients might be uneven. Techniques like oversampling, undersampling, or SMOTE (Synthetic Minority Oversampling Technique) can be employed to address this.

### **Dimensionality Reduction (PCA)**
Employs Principal Component Analysis (PCA) to potentially reduce the number of features while retaining important information. This can improve model performance and reduce overfitting.

## **Machine Learning Models**

Artificial Neural Network **(ANN)** Model (Keras)
Uses Keras, a deep learning library built on top of TensorFlow, to construct a Sequential model.

### **Overfitting Addressed**: The notebook likely incorporates techniques like regularization (L1/L2) or dropout layers to prevent overfitting in the ANN model.

### **Random Forest Classifier**
Implements a Random Forest model, known for its robustness to overfitting and handling imbalanced datasets.

##**Model Training and Evaluation**

**Training**: Trains both the ANN and Random Forest models on the prepared data.

**Evaluation**
Evaluates the performance of each model using metrics like accuracy, precision, recall, and F1-score. Include details on the achieved accuracy for both models on the training and test sets. The difference in accuracy between sets can indicate potential for improvement in generalizability.
