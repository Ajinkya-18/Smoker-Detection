# 🧠 Smoker Detection using Bio-signals

This project aims to classify individuals as **smokers or non-smokers** using a range of bio-signal parameters from annual health examination data. The analysis combines domain knowledge from health sciences with machine learning to uncover the most relevant features affected by smoking behavior.

---

## 📂 Project Overview

- **Task**: Binary classification (`smoker` vs `non-smoker`)
- **Data Source**: [Kaggle - Playground Series S3E24] (https://www.kaggle.com/competitions/playground-series-s3e24/data)
- **Input**:  Features from general health checkups
- **Target**:  `smoking` (1 = smoker, 0 = non-smoker)

---

## 📊 Features Used

The following features were selected after correlation analysis, mutual information checks, and domain relevance:

- `age`
- `fasting blood sugar`
- `triglyceride`
- `HDL` (High-density lipoprotein)
- `LDL` (Low-density lipoprotein)
- `hemoglobin`
- `serum creatinine`
- `AST` (Aspartate transaminase)
- `ALT` (Alanine transaminase)
- `Gtp` (Gamma-glutamyl transpeptidase)
- `dental caries`
- `scaled_BMI`
- `pulse_pressure` *(engineered feature = systolic - diastolic)*
- `smoking` *(target variable)*

---

## 🧪 Methods & Tools

- **Data Cleaning**
  - Outlier detection and handling
  - Missing value checks
- **Feature Engineering**
  - `pulse_pressure` and `scaled_BMI`
  - Robust scaling for outlier-resilient normalization
- **Modeling**
  - Exploratory Data Analysis (EDA)
  - Correlation and Mutual Information heatmaps
  - Classification using tree-based models (Random Forest, etc.)
- **Evaluation**
  - Accuracy, F1-score, Confusion Matrix
  - Feature importance ranking

---
## ⚙️ Environment Setup

You can set up the project using either Conda or `requirements.txt`:

### Option 1: Conda (recommended)
#### Create and activate environment from environment.yml
conda env create -f environment.yml
conda activate smoker-detection

### Option 2: pip and requirements.txt

#### Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

#### Install dependencies
pip install -r requirements.txt

#### Create environment from conda file
conda env create -f environment.yml
conda activate smoker-detection

## Results Summary
The model was able to detect smokers with moderate accuracy and interpretability. Feature importance analysis revealed that Gtp, ALT, and HDL are strong predictors, consistent with known medical indicators of smoking-related effects. More feature engineering and fine-tuning is necessary to further improve the models accuracies.
