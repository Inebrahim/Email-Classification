# Email Spam-Ham Classification

## Overview
This project focuses on implementing and comparing multiple **Machine Learning** (ML) models for **spam detection** in emails. The goal is to classify emails as either **Spam** or **Ham** (non-spam) using a variety of classification techniques, including **Support Vector Classifier (SVC)**, **Random Forest**, **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Decision Trees**. The project evaluates the models using performance metrics such as **accuracy** and **precision** to determine the best-performing algorithm.

## Key Features
- **Machine Learning Models**:
  - Support Vector Classifier (SVC)
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - J48 Decision Tree
- **Libraries Used**:
  - Python (Pandas, NumPy, Scikit-learn)
  - Machine Learning: SVC, Random Forest, Logistic Regression, KNN, Decision Trees
  - Natural Language Processing (NLP) for text preprocessing

- **Performance Metrics**:
  - **Accuracy**: Percentage of correctly classified emails.
  - **Precision**: Percentage of true positive predictions out of all positive predictions.
  - **Recall and F1-Score** (optional, but can be added for a more detailed evaluation).

## Models and Results

### 1. **Support Vector Classifier (SVC)**
- **Accuracy**: 98.93% (Best performing model)
- **Precision**: 1.0
- **Description**: SVC delivers excellent performance, with perfect precision, making it the best choice for this problem in terms of overall accuracy.

### 2. **Random Forest**
- **Accuracy**: 97.99%
- **Precision**: 1.0
- **Description**: Random Forest shows strong performance with perfect precision, making it a great candidate for spam detection.

### 3. **Logistic Regression**
- **Accuracy**: 96.92%
- **Precision**: 1.0
- **Description**: Logistic Regression also achieves high precision and accuracy, making it reliable for email classification.

### 4. **K-Nearest Neighbors (KNN)**
- **Accuracy**: 97.86%
- **Precision**: 0.968
- **Description**: KNN provides a balanced trade-off between accuracy and precision, offering solid performance for spam detection.

### 5. **Decision Tree**
- **Accuracy**: 95.62%
- **Precision**: 0.946
- **Description**: Decision Trees offer moderate performance, with a slightly lower precision compared to other models.

### 6. **J48 Decision Tree**
- **Accuracy**: 96.09%
- **Precision**: 0.875
- **Description**: J48 Decision Tree performs lower than other models in terms of both accuracy and precision but still provides a decent classification solution.

## Requirements
To run this project, you will need the following libraries:
- Python 3.x
- Scikit-learn
- Pandas
- NumPy

You can install all dependencies using pip:
```bash
pip install scikit-learn pandas numpy
```

## Dataset
The dataset used for spam-ham classification consists of labeled emails categorized as **Spam** or **Ham**. You can find the dataset [here](add-dataset-link-here). The dataset is preprocessed to extract useful features like email content, subject, and other textual data for model training.

## How to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/email-spam-ham-classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd email-spam-ham-classification
   ```

3. Prepare the dataset:
   - Place the dataset in the appropriate directory as required by the script (refer to the specific dataset structure in the project).

4. Run the model training script:
   ```bash
   python train_model.py
   ```

5. To test predictions with new data, use:
   ```bash
   python predict.py --input new_email_data.csv
   ```

## Evaluation Metrics
- **Accuracy**: Measures the overall correctness of the model (how many emails are correctly classified).
- **Precision**: Measures the proportion of true positive classifications among the emails predicted as spam.

## Acknowledgments
- The project uses **Scikit-learn** for machine learning models and **Pandas** and **NumPy** for data preprocessing.
- Dataset credits to the creators for providing the labeled email data for classification.
