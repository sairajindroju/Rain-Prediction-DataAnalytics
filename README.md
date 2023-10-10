# Rainfall Prediction Project README

## Table of Contents
1. [Business Understanding](#business-understanding)
   1.1 [Business Problem](#business-problem)
   1.2 [Dataset](#dataset)
   1.3 [Proposed Analytics Solution](#proposed-analytics-solution)
2. [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
   2.1 [Data Quality Report](#data-quality-report)
   2.2 [Missing Values and Outliers](#missing-values-and-outliers)
   2.3 [Normalization](#normalization)
   2.4 [Transformations](#transformations)
   2.5 [Feature Selection](#feature-selection)
3. [Model Selection](#model-selection)
   3.1 [Logistic Regression](#logistic-regression)
   3.2 [Random Forest Classifier](#random-forest-classifier)
   3.3 [KNN Classifier](#knn-classifier)
   3.4 [Naive Bayes Classifier](#naive-bayes-classifier)
   3.5 [AdaBoost Classifier](#adaboost-classifier)
   3.6 [Gradient Boosting Classifier](#gradient-boosting-classifier)
   3.7 [XGBoost Classifier](#xgboost-classifier)
4. [Evaluation](#evaluation)
   4.1 [Accuracy](#accuracy)
   4.2 [Sensitivity](#sensitivity)
   4.3 [Specificity](#specificity)
   4.4 [Precision Score](#precision-score)
   4.5 [False Negative Rate](#false-negative-rate)
   4.6 [Youden’s Index](#youdens-index)
   4.7 [Discriminant Power](#discriminant-power)
   4.8 [Balanced Classification Rate](#balanced-classification-rate)
   4.9 [Geometric Mean](#geometric-mean)
5. [Results](#results)

---

## 1. Business Understanding

### 1.1 Business Problem

Global warming is affecting ecosystems worldwide, and Australia is particularly vulnerable to the impacts of climate change, including rising temperatures, sea level rise, coral bleaching, and extreme weather events such as bushfires. One critical issue arising from these changes is food security, as agriculture relies heavily on rainfall. This project aims to predict whether it will rain in Australia the next day, with a focus on building budget-friendly rainfall forecast applications.

### 1.2 Dataset

The dataset used for this project was obtained from Kaggle and contains 23 features and 145,461 rows. The target variable is "RainTomorrow," which indicates whether it will rain the next day. Some of the features in the dataset include:
- Date
- Location (weather station name)
- Minimum and Maximum Temperature
- Rainfall
- Evaporation
- Sunshine hours
- Wind direction and speed
- Humidity
- Atmospheric pressure
- Cloud cover
- Temperature at different times of the day
- Rain today (binary)
- Rain tomorrow (target variable)

### 1.3 Proposed Analytics Solution

The analytics solution proposed for this project involves the following steps:

1. **Gathering Data:** Data was collected from various sources, and a Kaggle dataset with relevant features for rainfall prediction was selected.

2. **Data Analysis:** The dataset was analyzed to gain a better understanding of its content and identify important features and trends that can aid in model building.

3. **Data Preprocessing:** Data quality issues were addressed, including handling missing values through imputation, and outliers were identified and managed.

4. **Feature Selection:** Relevant features were selected for model building using techniques such as Chi-square test, PCA, and Recursive Feature Elimination (RFE).

## 2. Data Exploration and Preprocessing

### 2.1 Data Quality Report

The data quality report includes metrics for both categorical and continuous variables, such as counts, missing values, cardinality, and key statistics.

### 2.2 Missing Values and Outliers

Missing values were identified in several features and were handled through imputation. Outliers were detected using box plots and the Interquartile Range (IQR) method.

### 2.3 Normalization

Continuous features were normalized using Min-Max normalization to bring them within the range [0, 1].

### 2.4 Transformations

Categorical data were transformed into numerical data using one-hot encoding.

### 2.5 Feature Selection

Feature selection techniques such as Chi-square test, PCA, and RFE were used to identify and select the most relevant features for model building.

## 3. Model Selection

Various classification models were evaluated for their effectiveness in predicting rainfall. The following models were considered:

### 3.1 Logistic Regression

Logistic Regression was used to model the relationship between input variables and the target variable. It achieved an accuracy of 85.03% and was evaluated using various metrics.

### 3.2 Random Forest Classifier

Random Forest, a robust ensemble algorithm, achieved an accuracy of 78.11% and was evaluated for its performance.

### 3.3 KNN Classifier

The K-Nearest Neighbors (KNN) classifier achieved an accuracy of 79.23% and was assessed for its effectiveness.

### 3.4 Naive Bayes Classifier

The Naive Bayes classifier, which assumes a normal distribution, achieved an accuracy of 78.11% and was evaluated.

### 3.5 AdaBoost Classifier

AdaBoost, an ensemble technique, achieved an accuracy of 84.47% and underwent evaluation.

### 3.6 Gradient Boosting Classifier

The Gradient Boosting classifier achieved an accuracy of 84.62%, and its performance was assessed.

### 3.7 XGBoost Classifier

The XGBoost classifier, an advanced ensemble method, achieved an accuracy of 85.62% and was evaluated.

## 4. Evaluation

Various evaluation metrics were used to assess the performance of the models, including accuracy, sensitivity, specificity, precision score, false negative rate, Youden’s Index, discriminant power, balanced classification rate, and geometric mean.

## 5. Results

The results of the model evaluation are summarized in the table below:

| Model               | Accuracy | Sensitivity | Precision Score | False Negative Rate | Youden’s Index | Discrimination Power | Balanced Classification Rate | Geometric Mean |
|---------------------|----------|-------------|-----------------|----------------------|----------------|-----------------------|-----------------------------|----------------|
| Logistic Regression | 0.8503   | 0.72        | 0.79            | 0.13                 | 0.59           | 1.55                  | 0.79                        | 0.79           |
| Random Forest       | 0.7811   | 0.66        | 0.82            | 0.16                 | 0.64           | 1.67                  | 0.82                        | 0.82           |
| KNN Classifier      | 0.7923   | 0.64        | 0.74            | 0.16                 | 0.47           | 1.2
