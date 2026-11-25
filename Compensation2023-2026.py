import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv("combined_2023_2026.csv")
years = range(2023, 2027)


# Make it so that only valid selections are added
import CompareYears as cy
import statsmodels.api as sm
variables = ['Age', 'DevType', 'EdLevel', 'WorkExp', 'OrgSize', 'AISelect', 'Year']
categorical_variables = ['Age', 'DevType', 'EdLevel', 'OrgSize']
for col in categorical_variables:
    valid_selections = cy.FindWhatsInAll(df, years, col, False)
    df = df[df[col].isin(valid_selections)]
    print(valid_selections, "\n\n")


df["Year"] = df["Year"] - years[0]
original_mean = df["WorkExp"].mean()
df["WorkExp"] = df["WorkExp"] - original_mean

addingAll = " + ".join(f"C({var})" if var in categorical_variables else var for var in variables)
formula = "np.log(ConvertedCompYearlyInflation) ~ " + addingAll
model = sm.OLS.from_formula(formula, data=df).fit(cov_type="HC3")

print(model.summary())

# This tests the model by training on all years but the final year and testing the model based on that final year
# Ex: Observed Years 2020-2026
#   Training Data: 2020-2025
#   Test Data = 2026s
cy.GetTestResults(df, years, formula)

# Testing original model with a multitude of tests
# Primarily AI was used for these functions and tests
# Basic diagnostics
# VIF (group and ungrouped)
# Residual Diagnostic with plots
# Influential Observations
# Heteroskedasticity test
# Durbin-Watson statistic
# Wald z-tests (Individual and joint)
# Multiple testing correction
cy.AI_Tests(df, model, print_influential_instance_results=True, original_mean=original_mean)
