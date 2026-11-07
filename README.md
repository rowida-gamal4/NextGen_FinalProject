# Mobile Wallet Fraud Detection

A machine learning project that detects potentially fraudulent mobile wallet transactions using transaction details and account balance information to help reduce financial losses while minimizing false alarms.

## Live Application

Try the application: [NextGen Fraud Detection](https://nextgen-fraud-detection.streamlit.app/)

GitHub Repository: [NextGen_FinalProject](https://github.com/rowida-gamal4/NextGen_FinalProject)

Screnshots : in ScreenShots Folder in GitHub

## Model Performance

The models were evaluated on an independent test set containing 1,272,524 transactions, with 1,643 fraudulent transactions.

The majority-class Dummy Classifier was used as the baseline. Random Forest was selected as the best-performing model.

| Metric | Baseline (Dummy Classifier) | Best Model (Random Forest) |

| Accuracy | 99.87% | 99.96% |
| Precision | 0.00% | 76.21% |
| Recall | 0.00% | 99.45% |
| F1-score | 0.00% | 86.30% |
| PR-AUC | 0.0013 | 0.9934 |
| ROC-AUC | 0.5000 | 0.9990 |

Final Random Forest configuration:
- Number of estimators: 150
- Maximum depth: 20
- Minimum samples per leaf: 2
- Class weight: Balanced
- Classification threshold: 0.10

### Test Set Results

At the selected threshold of 0.10, the model achieved:

- True Positives: 1,634
- True Negatives: 1,270,371
- False Positives: 510
- False Negatives: 9

The model detected approximately 99.45% of fraudulent transactions in the test set.

The estimated business cost was 14,100, based on an assumed cost of 1,000 per missed fraud transaction and 10 per false alarm.

## Project Overview

### Dataset

This project uses the PaySim Mobile Money dataset, a synthetic dataset simulating mobile financial transactions.

The original dataset is not included in this repository because of its large size.

Dataset source: [PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Features Used

The model uses five original transaction features:

- 'step'
- 'type'
- 'amount'
- 'oldbalanceOrg'
- 'oldbalanceDest'

Additional features are generated through a custom feature engineering function, including transaction timing, logarithmic amount, balance ratios, and insufficient-origin-balance indicators.

Post-transaction balance features were excluded to avoid data leakage because the prediction is intended to happen before transaction completion.

### Machine Learning Workflow

1. Data understanding and exploratory data analysis.
2. Handling class imbalance through class weighting.
3. Feature selection and engineering.
4. Preprocessing using Scikit-learn pipelines.
5. Training and comparing multiple classification algorithms.
6. Cross-validation and hyperparameter tuning.
7. Selecting a classification threshold based on business costs.
8. Evaluating the final model on an untouched test set.
9. Building and deploying an interactive Streamlit application.

### Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## How to Run Locally

### 1. Clone the repository


git clone https://github.com/rowida-gamal4/NextGen_FinalProject.git


### 2. Install dependencies

Python 3.8 or a compatible environment is required for the pinned dependencies.

pip install -r requirements.txt


### 3. Run the application

streamlit run app.py

Open the local URL displayed in your terminal, usually:

http://localhost:8501

## Using the Application

The application provides two prediction modes:

1. Single Transaction
- Enter the transaction details.
- Submit the transaction.
- View the predicted fraud probability and classification.

2. CSV Upload
- Upload a CSV containing the five required input features.
- The application generates predictions for all transactions.
- If the CSV contains an 'isFraud' column, evaluation metrics are also displayed.
- Download the results as a CSV file.

## Limitations

- PaySim is a synthetic dataset, so its patterns may not fully represent real-world mobile wallet fraud.
- The dataset is highly imbalanced.
- False positives may require manual investigation.
- Business costs used for threshold selection are assumptions rather than actual financial losses.
- Performance on real-world transactions requires further validation.

## Conclusion

The final Random Forest model achieved a PR-AUC of 0.9934 and recall of 99.45% on the independent test set. The project demonstrates an end-to-end machine learning workflow, from data exploration and model development to an interactive, publicly deployed fraud detection application.
