# 🚢 Titanic Survival Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TQUucmP9gmaTNVu4WF0s-MPkU85aUZpN?usp=sharing)

A machine learning model to predict passenger survival on the Titanic using logistic regression. Built in Google Colab for easy access and reproducibility.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Key Steps](#-key-steps)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Future Work](#-future-work)
- [Dependencies](#-dependencies)
- [License](#-license)

## 🌟 Project Overview
This project analyzes the Titanic dataset to predict passenger survival using logistic regression. The workflow includes:
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Performance analysis

## 📁 Dataset
The dataset contains the following columns:
📌 Dataset size: 891 rows × 9 columns (after cleaning)
Dataset : Titanic-Dataset.csv ( From Kaggle "https://www.kaggle.com/datasets/yasserh/titanic-dataset" )

| Column | Description |
|--------|-------------|
| PassengerId | Unique identifier |
| Survived | Survival status (0 = No, 1 = Yes) |
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Passenger fare |
| Embarked | Port of embarkation |

## 🔑 Key Steps
1. **Data Preprocessing**:
   - Filled missing Age values with median
   - Dropped Cabin column (77% missing)
   - Encoded categorical features (Sex, Embarked)

2. **Model Development**:
   - Logistic Regression with 200 max iterations
   - 80-20 train-test split
   - Standardized numeric features

3. **Evaluation**:
   - Accuracy metrics
   - Confusion matrix
   - Feature importance analysis

## 📊 Results
The model achieved **81.01% accuracy**:

| Metric | Not Survived (0) | Survived (1) |
|--------|------------------|--------------|
| Precision | 0.83 | 0.79 |
| Recall | 0.86 | 0.74 |
| F1-score | 0.84 | 0.76 |

![Feature Importance](feature_importance.png)

## 🚀 Quick Start
Two ways to run:

### Option 1: Google Colab (Recommended)
1. Click the **Open in Colab** button above
2. Select "Runtime" → "Run all" (Ctrl+F9)
3. The notebook will execute all cells automatically

## Option 2: Local Execution
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
pip install -r requirements.txt
jupyter notebook Titanic_Survival_Prediction.ipynb
```

## 💡 Future Work:
- Experiment with Random Forest and XGBoost
- Add hyperparameter tuning cells
- Perform cross-validation
- Explore model interpretability with SHAP

## 📦 Dependencies:
- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn

## 📜 License:
This project is licensed under the MIT License.

## 👤 Author:
- GitHub: [Dharshini19960](https://github.com/Dharshini19960)
- LinkedIn: [Dharshini Ravilla](https://www.linkedin.com/in/dharshini-ravilla-610964313)
