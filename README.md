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
├── app
├── images
├── report
│   └── proposal.md
└── README.md
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
