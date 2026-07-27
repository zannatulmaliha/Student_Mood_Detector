# Student Mood Detector - Project Proposal

## 1. Objective
To develop a predictive model that can identify the risk of depression among students based on behavioral, academic, and physiological factors.

## 2. Dataset Overview
We are utilizing the Student Life/Depression Dataset. This dataset consists of student metrics designed to evaluate the stress and mood of individuals in an academic setting.

| Feature | Meaning | Type |
| :--- | :--- | :--- |
| Age | Student age | Numeric |
| Gender | Student gender | Categorical |
| Sleep Duration | Daily sleep | Numeric |
| Academic Pressure | Stress due to academics | Numeric |
| Depression | Target variable | Target |

## 3. Methodology
- **Data Collection:** Gather the student dataset containing key metrics (Age, Gender, Sleep Duration, Academic Pressure).
- **Data Exploration (EDA):** Use correlation heatmaps, histograms, and boxplots to understand feature distributions and correlations with the target.
- **Data Preprocessing:** 
  - Handle missing data (imputation).
  - Remove duplicates.
  - Encode categorical data (Gender).
  - Standardize numerical data (Age, Sleep Duration, Academic Pressure) using `StandardScaler`.
- **Modeling:** Train and evaluate classification models (e.g., Logistic Regression, Random Forest) to predict the binary target (`Depression`).
- **Evaluation:** Measure model performance using metrics such as Accuracy, Precision, Recall, and F1-score.

## 4. Expected Results
- A reliable predictive model capable of assessing depression risk based on basic student metrics.
- Clear insights (via visualizations) on how factors like *Academic Pressure* and *Sleep Duration* correlate with student mood.
- A well-documented data pipeline to allow future addition of features like app usage or mobility data.
