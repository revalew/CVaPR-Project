[https://app.readytensor.ai/publications/rr8X2XYGGLAa](https://app.readytensor.ai/publications/rr8X2XYGGLAa)


# Tumour Data Classification and Metastasis Prediction

---

## Project Overview

This project ([available on GitHub](https://github.com/revalew/CVaPR-Project/)), was part of the Computer Vision and Pattern Recognition (CVaPR) course, and focuses on the classification of medical data related to tumour metastasis. Using both simple classifiers and advanced neural networks, the goal was to develop a robust model to classify data into "metastasis" (class 0) or "no metastasis" (class 1). The project leverages anonymized data derived from real-world patient studies, ensuring compliance with privacy standards.

### Key Features of the Project

- Comparison of **simple classifiers** (SVM, KNN, Random Forest, Naive Bayes) with advanced models like CNNs.
- Utilization of **feature selection methods** to improve classification accuracy, including ranking, wrapper, and embedded methods.
- Implementation of **hyperparameter optimization** using the *optuna* library for the CNN model.
- Addressing challenges posed by **imbalanced classes** through careful dataset splitting and performance metric selection.

---


The original final report containing the results of the different classification methods (in Polish) is available [on GitHub](https://github.com/revalew/CVaPR-Project/blob/master/final_report/CVaPR_report.pdf). The translated report (in English) was added for this submission and is also available [on GitHub](https://github.com/revalew/CVaPR-Project/blob/master/final_report/CVaPR_report(EN).pdf) (the report was translated quickly due to limited time and may contain some errors).

---

## File Structure

### `/src/` Directory

Contains key scripts and notebooks for building and evaluating classifiers:

1. **`cnn_classification.ipynb`**: Implements a Convolutional Neural Network with optimized hyperparameters.
2. **`simple_classifier_knn_svm_bayes.ipynb`**: Implements simple classifiers, including:
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)
   - Random Forest
   - Naive Bayes
   - Decision Tree
3. **`dense_network_classification.ipynb`**: Explores a Dense Neural Network as a baseline for comparison.
4. **`xtrain_feature_selection/`**: Scripts for feature selection on training data:
   - `ranking_method.ipynb`: Assesses individual feature impact.
   - `wrapper_method.ipynb`: Tests combinations of features.
   - `embedded_method.ipynb`: Selects features during training.
   - `simple_classifier_ranking_method.ipynb`: Applies ranking to simple classifiers.
5. **`all_feature_selection/`**: Tests feature selection methods on the full dataset.
6. **`optimizers/`**: Contains scripts for hyperparameter tuning:
   - `optimalization_test.ipynb`
   - `hiperparametr_optymalization.ipynb`
   - `learning_optimalization.ipynb`

### `/models/` Directory

Stores trained models, including:
- CNN and DNN models.
- Models generated using feature selection methods (`xtrain_feature_selection/` and `all_feature_selection/`).
- Optimized models using *optuna* (`optimizers/`).

### `/data/` Directory

Houses all data used for training and testing:
- **`labels_features.csv`**: Combines features and labels.
- **`features.csv`**: Contains features only.
- **`labels.csv`**: Contains labels only.
- **`clinical_radiomics_imported_from_tsv.xlsx`**: The original dataset.

---

## Methodologies and Insights

### Simple Classifiers

- **SVM, KNN, Random Forest, and Naive Bayes** classifiers were implemented to establish baselines. These models were further refined using feature selection methods.

### Convolutional Neural Network (CNN)
- A CNN was constructed with hyperparameters optimized via *optuna*. Layers were fine-tuned to extract meaningful patterns from the dataset.

### Feature Selection
Feature selection was critical for reducing dimensionality and enhancing model performance:
- **Ranking Method**: Selected features based on individual importance scores.
- **Wrapper Method**: Evaluated feature subsets iteratively.
- **Embedded Method**: Incorporated feature selection within model training.

### Optimization
- Hyperparameter optimization using *optuna* significantly improved the CNN's accuracy.
- Experiments explored the balance between model complexity and overfitting.

---

## Classifiers and Neural Networks in Tumour Data Classification

### Simple Classifiers

#### 1. **Support Vector Machine (SVM)**
SVMs map input data to a high-dimensional feature space, making it possible to separate data points using a hyperplane. They are effective for both linear and non-linear classification, regression tasks, and outlier detection.

#### 2. **K-Nearest Neighbors (KNN)**
KNN classifies data based on the closest training samples in the feature space. The algorithm determines the class of a data point by majority voting among its `k` nearest neighbors, with proximity measured using distance metrics.

#### 3. **Random Forest**
Random Forest is an ensemble learning method that builds multiple decision trees on random subsets of data and features. The final classification is determined by majority voting among the trees, offering robustness and accuracy.

#### 4. **Naive Bayes**
This probabilistic classifier is based on Bayes' theorem, assuming feature independence. It calculates the probability of each class for a given data point and assigns it to the class with the highest probability.

### Neural Networks

#### **Convolutional Neural Networks (CNNs)**
CNNs are specialized neural networks designed for processing structured data like images. They consist of convolutional layers that extract local patterns, such as edges and textures, followed by pooling layers that reduce dimensionality. The features are then passed to fully connected layers for classification. CNNs are particularly effective in handling complex patterns and features, making them suitable for medical data analysis.

#### Key Advantages of CNNs in the Project
- **Feature Extraction**: Automatically identifies relevant patterns without manual feature engineering.
- **Hyperparameter Optimization**: Utilized the *optuna* library to fine-tune parameters, improving performance.
- **Handling Imbalance**: Combined with stratified sampling and evaluation metrics (e.g., F1-score) to manage class imbalance.

#### **Dense Neural Network (DNN)**
A simpler type of neural network where every neuron in one layer is connected to every neuron in the next. DNNs were tested as a baseline but lacked the specialized feature extraction capabilities of CNNs.

This combination of simple classifiers and advanced neural networks provided a comprehensive evaluation of the classification task, balancing interpretability and accuracy.

---

## Challenges and Solutions

- **Imbalanced Dataset**: Addressed using stratified splitting and F1-score as a performance metric.
- **Feature Redundancy**: Eliminated through rigorous selection techniques.
- **Model Instability**: Mitigated by cross-validation and parameter tuning.

---

## Conclusion

This project demonstrated the importance of combining advanced classification models with effective feature selection and hyperparameter optimization. The use of anonymized real-world data ensures the results are both impactful and ethically sound.