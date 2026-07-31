# Student Mood Detector

A project to analyze and detect student depression based on academic and behavioral factors.

## Project Structure

```text
Student_Mood_Detector/
│
├── data
│   └── student_dataset.csv
├── notebook
│   ├── Phase_3_Exploration.ipynb
│   └── Phase_4_Cleaning.ipynb
├── models
│   └── mood_model.joblib
├── train_model.py
├── app.py
├── report
│   └── proposal.md
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python train_model.py   # trains the model, saves models/mood_model.joblib
streamlit run app.py    # launches the web app
```

## Dataset Information

| Feature | Meaning | Type |
| :--- | :--- | :--- |
| Age | Student age | Numeric |
| Gender | Student gender | Categorical |
| Sleep Duration | Daily sleep | Numeric |
| Academic Pressure | Stress due to academics | Numeric |
| Depression | Target variable | Target |

*Note: The dataset was downloaded/synthesized as requested, containing 1000+ entries corresponding to the features above.*

## Phases Completed

- **Phase 1:** Created the repository and folder structure.
- **Phase 2:** Dataset downloaded and structured. Added feature description table.
- **Phase 3:** Notebook created for EDA with pandas operations and visualizations (Histograms, Boxplots, Correlation Heatmap).
- **Phase 4:** Data cleaning notebook created (handling duplicates, missing values, encoding, scaling, and train/test splits).
- **Phase 5:** Proposal updated in `report/proposal.md` based on the new dataset features.
- **Phase 6:** `train_model.py` builds a scikit-learn pipeline (median/impute + scale + one-hot encode + RandomForest, `class_weight="balanced"` for the imbalanced target) and saves it to `models/mood_model.joblib`.
- **Phase 7:** `app.py` is a Streamlit app that loads the saved model and predicts depression risk from user-entered Age, Gender, Sleep Duration, and Academic Pressure.
