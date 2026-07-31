"""Phase 6: Train and compare classifiers, save the best one for app.py."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

NUM_COLS = ["Age", "Sleep Duration", "Academic Pressure"]
CAT_COLS = ["Gender"]

df = pd.read_csv("data/student_dataset.csv").drop_duplicates()
X = df[NUM_COLS + CAT_COLS]
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ]
)

CANDIDATES = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=42),
}

results = []
best_name, best_model, best_f1 = None, None, -1

for name, clf in CANDIDATES.items():
    model = Pipeline([("preprocess", preprocess), ("clf", clf)])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})
    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print("Confusion Matrix:\n", cm)

    if f1 > best_f1:
        best_name, best_model, best_f1 = name, model, f1

print("\n=== Comparison ===")
print(pd.DataFrame(results).set_index("Model").round(3))

print(f"\nBest model: {best_name} (F1-score={best_f1:.3f})")
joblib.dump(best_model, "models/mood_model.joblib")
print("Saved models/mood_model.joblib")
