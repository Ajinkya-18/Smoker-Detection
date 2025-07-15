🧠 Smoker Detection using Bio-signals
This project addresses the binary classification task of identifying individuals as smokers or non-smokers based on routine health examination data. By integrating domain knowledge from biomedical sciences with machine learning and model interpretability tools, the goal is to uncover physiological markers most indicative of smoking behavior.

---

## 📂 Project Overview

- **Objective**: Binary classification (`smoker` vs `non-smoker`)
- **Data Source**: [Kaggle - Playground Series S3E24] (https://www.kaggle.com/competitions/playground-series-s3e24/data)
- **Input**:  Features from general health checkups (Health-related biosignal parameters and derived biomarkers)
- **Target**:  `smoking` (1 = smoker, 0 = non-smoker)

---

📊 Features Used
Following correlation analysis, recursive feature elimination (RFECV), SHAP value explanations, and domain-relevance screening, the final features used were:

age

systolic

relaxation (diastolic blood pressure)

fasting blood sugar

cholesterol

triglyceride

HDL (High-density lipoprotein)

LDL (Low-density lipoprotein)

hemoglobin

urine protein

serum creatinine

AST (Aspartate transaminase)

ALT (Alanine transaminase)

Gtp (Gamma-glutamyl transpeptidase)

dental caries

BMI

WHtR (waist-to-height ratio)

eyesight

hearing (retained after feature importance analysis)

⚙️ Methods & Pipeline
🔧 Preprocessing
Missing Value Handling: Checked and confirmed complete records.

Outlier Normalization: QuantileTransformer used for outlier-heavy distributions.

Feature Engineering:

Engineered metrics: pulse_pressure = systolic - diastolic

scaled_BMI and other ratio-based features explored

Feature Selection:

RFECV with LogisticRegression, DecisionTreeClassifier, and RandomForestClassifier

Permutation importance and SHAP value inspection

Domain-guided selection refinement

🤖 Models Used
Logistic Regression (L1/L2 regularization) – interpretable baseline

Decision Tree Classifier – fast feature selector and benchmark model

Random Forest Classifier – ensemble model with best overall performance

📈 Evaluation Metrics
Accuracy

F1 Score

ROC-AUC Curve

SHAP Analysis for interpretability

Confusion Matrix

📊 Results Summary
Model	F1 Score	AUC Score
Logistic Regression	0.70	0.83
Random Forest Classifier	0.75	0.85
Decision Tree Classifier	~0.72	~0.82

Top Predictive Features (SHAP + Tree Importance):

Gtp, ALT, HDL, LDL, hemoglobin, and age consistently ranked high across models.

Observations:

RandomForestClassifier demonstrated better generalization but favored conservative predictions.

LogisticRegression had better balance in sensitivity (smoker detection) and interpretability.

Tree-based models leaned toward non-smoker predictions in ambiguous cases, highlighting class imbalance handling requirements.

📊 SHAP Analysis
Used shap.Explainer for all models.

Visualized feature impacts using beeswarm plots.

Logistic Regression showed clearer separation and wider SHAP value ranges.

Tree models exhibited localized influence with conservative thresholds.

🖥️ Environment Setup
📁 Option 1: Conda (Recommended)
bash
Copy
Edit
conda env create -f environment.yml
conda activate smoker-detection
📁 Option 2: Pip + venv
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Unix/Mac

pip install -r requirements.txt
📌 Future Work
Address slight class imbalance (~70K smokers vs ~90K non-smokers)

Explore non-linear models (e.g., XGBoost, LightGBM)

Apply more granular feature engineering and ensemble strategies

Incorporate temporal analysis if multi-year checkup data becomes available

