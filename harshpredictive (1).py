# ============================================================
# GROUNDWATER LEVEL ANALYSIS AND PREDICTION USING MACHINE LEARNING
# Dataset: CGWB Groundwater Level Quarterly Manual
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\Dell_Owner\Downloads\gwl_manual_quarterly_cgwb_an_1991_2020.csv",
    low_memory=False
)

print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Replace dash placeholders with missing values
df.replace(r"^\s*-\s*$", np.nan, regex=True, inplace=True)

# Define target
target = "Groundwater Level Quarterly Manual (meter)"

# Convert date column
df["Data Acquisition Time"] = pd.to_datetime(
    df["Data Acquisition Time"],
    dayfirst=True,
    errors="coerce"
)

# Convert numerical columns
numeric_columns = [
    "Latitude",
    "Longitude",
    "RL_MSL",
    target
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate records
duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

df.drop_duplicates(inplace=True)

# Remove records without target or date
df.dropna(
    subset=[target, "Data Acquisition Time"],
    inplace=True
)

# Remove completely empty columns
df.dropna(axis=1, how="all", inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# Groundwater level distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    df[target],
    kde=True
)

plt.title("Groundwater Level Distribution")
plt.xlabel("Groundwater Level (Meter)")
plt.ylabel("Frequency")
plt.show()


# Boxplot

plt.figure(figsize=(10, 5))

sns.boxplot(x=df[target])

plt.title("Groundwater Level Outlier Detection")
plt.xlabel("Groundwater Level (Meter)")
plt.show()


# Groundwater level over time

plt.figure(figsize=(14, 5))

plt.scatter(
    df["Data Acquisition Time"],
    df[target],
    alpha=0.6
)

plt.title("Groundwater Level Over Time")
plt.xlabel("Date")
plt.ylabel("Groundwater Level (Meter)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Correlation heatmap

corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nCorrelation With Groundwater Level:")
print(corr[target].sort_values(ascending=False))


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

df["Year"] = df["Data Acquisition Time"].dt.year
df["Month"] = df["Data Acquisition Time"].dt.month
df["Quarter"] = df["Data Acquisition Time"].dt.quarter
df["DayOfYear"] = df["Data Acquisition Time"].dt.dayofyear

# Sort chronologically
df.sort_values(
    by="Data Acquisition Time",
    inplace=True
)

df.reset_index(drop=True, inplace=True)

print("\nDate Range:")
print(df["Data Acquisition Time"].min())
print(df["Data Acquisition Time"].max())


# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

numeric_features = [
    "Latitude",
    "Longitude",
    "Year",
    "Month",
    "Quarter",
    "DayOfYear"
]

categorical_features = [
    "Station",
    "District",
    "Tehsil",
    "Block"
]

numeric_features = [
    col for col in numeric_features
    if col in df.columns
]

categorical_features = [
    col for col in categorical_features
    if col in df.columns
]

features = numeric_features + categorical_features

X = df[features]
y = df[target]

# Chronological 80:20 train-test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 7. MODEL 1: SIMPLE LINEAR REGRESSION
# ============================================================

simple_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

simple_model.fit(
    X_train[["Latitude"]],
    y_train
)

simple_pred = simple_model.predict(
    X_test[["Latitude"]]
)

simple_mae = mean_absolute_error(y_test, simple_pred)
simple_mse = mean_squared_error(y_test, simple_pred)
simple_rmse = np.sqrt(simple_mse)
simple_r2 = r2_score(y_test, simple_pred)

print("\n========== SIMPLE LINEAR REGRESSION ==========")

print("MAE:", simple_mae)
print("MSE:", simple_mse)
print("RMSE:", simple_rmse)
print("R2 Score:", simple_r2)


# ============================================================
# 8. MODEL 2: MULTIPLE LINEAR REGRESSION
# ============================================================

multiple_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

multiple_model.fit(X_train, y_train)

multiple_pred = multiple_model.predict(X_test)

multiple_mae = mean_absolute_error(y_test, multiple_pred)
multiple_mse = mean_squared_error(y_test, multiple_pred)
multiple_rmse = np.sqrt(multiple_mse)
multiple_r2 = r2_score(y_test, multiple_pred)

print("\n========== MULTIPLE LINEAR REGRESSION ==========")

print("MAE:", multiple_mae)
print("MSE:", multiple_mse)
print("RMSE:", multiple_rmse)
print("R2 Score:", multiple_r2)


# ============================================================
# 9. MODEL 3: POLYNOMIAL REGRESSION
# ============================================================

poly_numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    ("scaler", StandardScaler())
])

poly_preprocessor = ColumnTransformer(
    transformers=[
        ("num", poly_numeric, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

poly_model = Pipeline(steps=[
    ("preprocessor", poly_preprocessor),
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)

poly_pred = poly_model.predict(X_test)

poly_mae = mean_absolute_error(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, poly_pred)

print("\n========== POLYNOMIAL REGRESSION ==========")

print("MAE:", poly_mae)
print("MSE:", poly_mse)
print("RMSE:", poly_rmse)
print("R2 Score:", poly_r2)


# ============================================================
# 10. REGRESSION MODEL COMPARISON
# ============================================================

regression_results = pd.DataFrame({

    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression"
    ],

    "MAE": [
        simple_mae,
        multiple_mae,
        poly_mae
    ],

    "MSE": [
        simple_mse,
        multiple_mse,
        poly_mse
    ],

    "RMSE": [
        simple_rmse,
        multiple_rmse,
        poly_rmse
    ],

    "R2 Score": [
        simple_r2,
        multiple_r2,
        poly_r2
    ]
})

print("\n========== REGRESSION COMPARISON ==========")

print(regression_results.to_string(index=False))


# Regression comparison graph

regression_results.set_index("Model")[
    ["MAE", "RMSE", "R2 Score"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Regression Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# Actual vs predicted

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    multiple_pred,
    alpha=0.7
)

min_value = min(y_test.min(), multiple_pred.min())
max_value = max(y_test.max(), multiple_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--"
)

plt.xlabel("Actual Groundwater Level")
plt.ylabel("Predicted Groundwater Level")
plt.title("Actual vs Predicted Groundwater Level")
plt.show()


# ============================================================
# 11. CLASSIFICATION TARGET CREATION
# ============================================================

# Create 3 relative categories using training data only

q1 = y_train.quantile(1/3)
q2 = y_train.quantile(2/3)

class_names = ["Shallow", "Moderate", "Deep"]

def classify_groundwater(value):

    if value <= q1:
        return "Shallow"

    elif value <= q2:
        return "Moderate"

    else:
        return "Deep"


y_train_class = y_train.apply(classify_groundwater)
y_test_class = y_test.apply(classify_groundwater)

print("\n========== CLASS DISTRIBUTION ==========")

print(y_train_class.value_counts())
print(y_test_class.value_counts())


# ============================================================
# 12. MODEL 4: DECISION TREE CLASSIFIER
# ============================================================

dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    ))
])

dt_model.fit(X_train, y_train_class)

dt_pred = dt_model.predict(X_test)

dt_prob = dt_model.predict_proba(X_test)

print("\n========== DECISION TREE CLASSIFIER ==========")

print("Accuracy:",
      accuracy_score(y_test_class, dt_pred))

print("Precision:",
      precision_score(
          y_test_class,
          dt_pred,
          average="macro",
          zero_division=0
      ))

print("Recall:",
      recall_score(
          y_test_class,
          dt_pred,
          average="macro",
          zero_division=0
      ))

print("F1 Score:",
      f1_score(
          y_test_class,
          dt_pred,
          average="macro",
          zero_division=0
      ))

print("Log Loss:",
      log_loss(
          y_test_class,
          dt_prob,
          labels=dt_model.named_steps["classifier"].classes_
      ))

print("\nClassification Report:")
print(classification_report(
    y_test_class,
    dt_pred,
    labels=class_names,
    zero_division=0
))


# Decision Tree Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test_class,
    dt_pred,
    labels=class_names,
    display_labels=class_names,
    cmap="Blues"
)

plt.title("Decision Tree Confusion Matrix")
plt.show()


# ============================================================
# 13. MODEL 5: SUPPORT VECTOR MACHINE
# ============================================================

svm_model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

svm_model.fit(X_train, y_train_class)

svm_pred = svm_model.predict(X_test)

svm_prob = svm_model.predict_proba(X_test)

print("\n========== SUPPORT VECTOR MACHINE ==========")

print("Accuracy:",
      accuracy_score(y_test_class, svm_pred))

print("Precision:",
      precision_score(
          y_test_class,
          svm_pred,
          average="macro",
          zero_division=0
      ))

print("Recall:",
      recall_score(
          y_test_class,
          svm_pred,
          average="macro",
          zero_division=0
      ))

print("F1 Score:",
      f1_score(
          y_test_class,
          svm_pred,
          average="macro",
          zero_division=0
      ))

print("Log Loss:",
      log_loss(
          y_test_class,
          svm_prob,
          labels=svm_model.named_steps["classifier"].classes_
      ))

print("\nClassification Report:")
print(classification_report(
    y_test_class,
    svm_pred,
    labels=class_names,
    zero_division=0
))


# SVM Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test_class,
    svm_pred,
    labels=class_names,
    display_labels=class_names,
    cmap="Greens"
)

plt.title("SVM Confusion Matrix")
plt.show()


# ============================================================
# 14. ROC-AUC EVALUATION
# ============================================================

# Multiclass One-vs-Rest ROC-AUC

print("\n========== ROC-AUC RESULTS ==========")

def calculate_auc(y_true, probabilities, model_name, model_classes):

    try:
        auc_value = roc_auc_score(
            y_true,
            probabilities,
            labels=model_classes,
            multi_class="ovr",
            average="macro"
        )

        print(model_name, "ROC-AUC:", auc_value)

        return auc_value

    except ValueError as error:

        print(model_name, "ROC-AUC unavailable:", error)

        return np.nan


dt_auc = calculate_auc(
    y_test_class,
    dt_prob,
    "Decision Tree",
    dt_model.named_steps["classifier"].classes_
)

svm_auc = calculate_auc(
    y_test_class,
    svm_prob,
    "SVM",
    svm_model.named_steps["classifier"].classes_
)


# ============================================================
# 15. CLASSIFICATION MODEL COMPARISON
# ============================================================

classification_results = pd.DataFrame({

    "Model": [
        "Decision Tree",
        "Support Vector Machine"
    ],

    "Accuracy": [
        accuracy_score(y_test_class, dt_pred),
        accuracy_score(y_test_class, svm_pred)
    ],

    "Precision": [
        precision_score(
            y_test_class, dt_pred,
            average="macro", zero_division=0
        ),
        precision_score(
            y_test_class, svm_pred,
            average="macro", zero_division=0
        )
    ],

    "Recall": [
        recall_score(
            y_test_class, dt_pred,
            average="macro", zero_division=0
        ),
        recall_score(
            y_test_class, svm_pred,
            average="macro", zero_division=0
        )
    ],

    "F1 Score": [
        f1_score(
            y_test_class, dt_pred,
            average="macro", zero_division=0
        ),
        f1_score(
            y_test_class, svm_pred,
            average="macro", zero_division=0
        )
    ],

    "Log Loss": [
        log_loss(
            y_test_class,
            dt_prob,
            labels=dt_model.named_steps["classifier"].classes_
        ),
        log_loss(
            y_test_class,
            svm_prob,
            labels=svm_model.named_steps["classifier"].classes_
        )
    ],

    "ROC-AUC": [
        dt_auc,
        svm_auc
    ]

})

print("\n========== CLASSIFICATION COMPARISON ==========")

print(classification_results.to_string(index=False))


# Classification comparison graph

classification_results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 16. FINAL MODEL COMPARISON
# ============================================================

print("\n========== FINAL RESULTS ==========")

print("\nREGRESSION RESULTS:")
print(regression_results.to_string(index=False))

print("\nCLASSIFICATION RESULTS:")
print(classification_results.to_string(index=False))


# Best regression model

best_regression = regression_results.loc[
    regression_results["RMSE"].idxmin()
]

print("\nBest Regression Model:")
print(best_regression.to_string())


# Best classification model

best_classification = classification_results.loc[
    classification_results["F1 Score"].idxmax()
]

print("\nBest Classification Model:")
print(best_classification.to_string())


# ============================================================
# 17. SAVE RESULTS
# ============================================================

# Save cleaned dataset

df.to_csv(
    "Cleaned_Groundwater_Dataset.csv",
    index=False
)

# Save regression results

regression_results.to_csv(
    "Regression_Model_Results.csv",
    index=False
)

# Save classification results

classification_results.to_csv(
    "Classification_Model_Results.csv",
    index=False
)

# Save predictions

predictions = pd.DataFrame({
    "Actual_Groundwater_Level": y_test.values,
    "Simple_Linear_Regression": simple_pred,
    "Multiple_Linear_Regression": multiple_pred,
    "Polynomial_Regression": poly_pred,
    "Actual_Class": y_test_class.values,
    "Decision_Tree_Prediction": dt_pred,
    "SVM_Prediction": svm_pred
})

predictions.to_csv(
    "Groundwater_Predictions.csv",
    index=False
)

print("\nAll datasets, results and predictions saved successfully!")

print("\n========== PROJECT COMPLETED ==========")
