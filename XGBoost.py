import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("combined_2023_2026.csv")
years = range(2023, 2027)

# Make it so that only valid selections are added
import CompareYears as cy
variables = ['Age', 'DevType', 'EdLevel', 'WorkExp', 'OrgSize', 'AISelect', 'Year']
categorical_variables = ['Age', 'DevType', 'EdLevel', 'OrgSize']
for col in categorical_variables:
    valid_selections = cy.FindWhatsInAll(df, years, col, False)
    df = df[df[col].isin(valid_selections)]
    # print(valid_selections, "\n\n")


df["Year"] = df["Year"] - years[0]
original_mean = df["WorkExp"].mean()
df["WorkExp"] = df["WorkExp"] - original_mean

addingAll = " + ".join(f"C({var})" if var in categorical_variables else var for var in variables)
formula = "np.log(ConvertedCompYearlyInflation) ~ " + addingAll

features = [
    "Age",
    "DevType",
    "EdLevel",
    "WorkExp",
    "OrgSize",
    "AISelect",
    "Year",
    "Industry"
]

target = "ConvertedCompYearlyInflation"

data = df.dropna(subset=[target, "DevType", "WorkExp"]).copy()

data = data[data[target] > 0]

categorical_features = [
    "Age",
    "DevType",
    "EdLevel",
    "OrgSize",
    "AISelect",
    "Industry"
]

numeric_features = [
    "WorkExp",
    "Year"
]

for col in categorical_features:
    data[col] = data[col].fillna("Missing")

data["WorkExp"] = pd.to_numeric(
    data["WorkExp"],
    errors="coerce"
)

data["WorkExp"] = data["WorkExp"].fillna(
    data["WorkExp"].median()
)

data["log_salary"] = np.log1p(data[target])

# Time-based split
train = data[data["Year"] < years[-1]-years[0]]
test = data[data["Year"] == years[-1]-years[0]]

X_train = train[features]
y_train = train["log_salary"]

X_test = test[features]
y_test = test["log_salary"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=12
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print("Log R-squared:",
      r2_score(y_test, predictions))

print("Log MAE:",
      mean_absolute_error(y_test, predictions))

print("Log RMSE:",
      np.sqrt(mean_squared_error(y_test, predictions)))

# Convert predictions back to salary
actual_salary = np.expm1(y_test)
predicted_salary = np.expm1(predictions)

print("Salary MAE:",
      mean_absolute_error(
          actual_salary,
          predicted_salary
      ))