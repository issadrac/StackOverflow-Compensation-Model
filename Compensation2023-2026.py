import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv("combined_2019_2026.csv")

years = range(2023, 2027)


df = df[df["Year"].isin(years)]
df = df[df["Country"] == "United States of America"]
# 0 values breaks the model because of np.log
# Confirmed that it was most likely mistakes given how few results there were espeically since for most roles, 
# the exact same number is the same for 500, 1000, 3000, 5000, 10000
df = df[df["ConvertedCompYearly"] > 10000]

# Convert AISelect to 0/1
df["AISelect"] = (~(df["AISelect"].str.contains("No,") | (df["AISelect"].str.contains("I don") & ~df["AISelect"].str.contains(";")))).astype("int8")
priority_indices = [
    474697, 514999, 532604, 518627,
    522691, 525723, 310371, 339534,
    371381, 492730, 536853, 527930,
    543718, 310821, 342951, 323760
]

print(
    df.loc[
        priority_indices,
        [
            "Year",
            "ConvertedCompYearly",
            "Age",
            "EdLevel",
            "WorkExp",
            "OrgSize",
            "DevType",
            "AISelect"
        ]
    ].to_string()
)

df.to_csv("BeforeRemovalCompensationDataFrame.csv")
# 474697: Academic researcher, primary education, 2 years experience, $9 million.
# 518627: Designer, 11 years experience, $1.91 million. (Most likely meant 200k not 2million originally)
# 522691: Back-end developer, 16 years experience, $28.7 million.
# 310821: Data scientist, 2 years experience, $9.51 million.
# 514999: 65 years or older, Senior Executive, 10k with 54 years of work experience
# 342951: Project Manager 11k with 38 years of working experience employed full time
df = df.drop(
    index=[474697, 518627, 522691, 310821, 514999, 342951],
    errors="ignore"
).copy()


df.to_csv("FinalCompensationDataFrame.csv")

# Make it so that only valid selections are added
import RelevantNow.Helpers.CompareYears as cy
import statsmodels.api as sm
variables = ['Age', 'DevType', 'EdLevel', 'WorkExp', 'OrgSize', 'AISelect', 'Year']
categorical_variables = ['Age', 'DevType', 'EdLevel', 'OrgSize']
for col in categorical_variables:
    valid_selections = cy.FindWhatsInAll(df, years, col, False)
    df = df[df[col].isin(valid_selections)]
    print(valid_selections, "\n\n")


df["Year"] = df["Year"] - years[0]
df["WorkExp"] = df["WorkExp"] - df["WorkExp"].mean()

addingAll = ""
for var in variables:
    if var in categorical_variables:
        addingAll += "C(" + var + ") + "
        if var != 'Year' and var != 'DevType':
            addingAll += "Year:C(" + var + ") + "
    else:
        addingAll += var + " + "
        if var != 'Year':
            addingAll += "Year:" + var + " + "           
addingAll = addingAll[0:len(addingAll)-3]
formula = "np.log(ConvertedCompYearly) ~ " + addingAll
model = sm.OLS.from_formula(formula, data=df).fit(cov_type="HC3")

print(model.summary())

def TestResults(df):
    import statsmodels.formula.api as smf
    
    # Split the data
    train_df = df[df["Year"] < 3].copy()
    test_df = df[df["Year"] == 3].copy()
    
    # Fit your original model on training data only
    model_test = smf.ols(
        formula=formula,
        data=train_df
    ).fit(cov_type="HC3")
    
    # Predict log compensation on unseen data
    y_actual = np.log(test_df["ConvertedCompYearly"])
    y_pred = model_test.predict(test_df)
    
    # Calculate prediction metrics
    errors = y_actual - y_pred
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_actual - y_actual.mean())**2)
    test_r2 = 1 - ss_res / ss_tot
    
    print()
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Test MAE:       {mae:.4f}")
    print(f"Test RMSE:      {rmse:.4f}")
    print()
    # Baseline: predict the training-set mean for everyone
    baseline_pred = np.repeat(
        np.log(train_df["ConvertedCompYearly"]).mean(),
        len(test_df)
    )
    
    y_actual = np.log(test_df["ConvertedCompYearly"])
    
    baseline_mae = np.mean(np.abs(y_actual - baseline_pred))
    baseline_rmse = np.sqrt(np.mean((y_actual - baseline_pred)**2))
    
    print("Baseline MAE:", baseline_mae)
    print("Model MAE:", mae)
    print("Percentage:", 100*(1-mae/baseline_mae))
    print()
    print("Baseline RMSE:", baseline_rmse)
    print("Model RMSE:", rmse)
    print("Percentage:", 100*(1-rmse/baseline_rmse))
    print()

TestResults(df)

# -----------------------------------------------------------------------------------------------------
# AI TESTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


# ============================================================
# 1. Basic model diagnostics
# ============================================================

X = model.model.exog
y = model.model.endog
n, p = X.shape

print("========== BASIC DIAGNOSTICS ==========")
print("Observations:", n)
print("Design matrix columns:", p)
print("Matrix rank:", np.linalg.matrix_rank(X))
print("Condition number:", np.linalg.cond(X))
print("R-squared:", model.rsquared)
print("Adjusted R-squared:", model.rsquared_adj)
print("AIC:", model.aic)
print("BIC:", model.bic)


# ============================================================
# 2. VIF for individual design-matrix columns
# ============================================================

print("\n========== INDIVIDUAL VIF VALUES ==========")

vif_results = []

for i, column_name in enumerate(model.model.exog_names):
    if column_name == "Intercept":
        continue

    try:
        vif_value = variance_inflation_factor(X, i)
    except Exception:
        vif_value = np.inf

    vif_results.append({
        "column": column_name,
        "VIF": vif_value
    })

vif_df = pd.DataFrame(vif_results).sort_values(
    "VIF",
    ascending=False
)

print(vif_df.to_string(index=False))


# ============================================================
# 3. VIF grouped by model term
#    This is more useful with categorical variables
# ============================================================

print("\n========== TERM-LEVEL VIF VALUES ==========")

import patsy

# Recreate the design matrix from the model formula
y_design, x_design = patsy.dmatrices(
    model.model.formula,
    model.model.data.frame,
    return_type="dataframe"
)
design_info = x_design.design_info
term_vif_results = []
for term_name, column_slice in design_info.term_name_slices.items():

    if term_name == "Intercept":
        continue

    term_columns = x_design.columns[column_slice]

    term_vifs = vif_df[
        vif_df["column"].isin(term_columns)
    ]["VIF"]

    term_vif_results.append({
        "term": term_name,
        "maximum_VIF": term_vifs.max(),
        "average_VIF": term_vifs.mean(),
        "number_of_columns": len(term_columns)
    })

term_vif_df = pd.DataFrame(term_vif_results).sort_values(
    "maximum_VIF",
    ascending=False
)

print(term_vif_df.to_string(index=False))


# ============================================================
# 4. Residual diagnostics
# ============================================================

residuals = model.resid
fitted_values = model.fittedvalues

influence = model.get_influence()

studentized_residuals = influence.resid_studentized_internal
leverage = influence.hat_matrix_diag
cooks_distance = influence.cooks_distance[0]

diagnostics_df = pd.DataFrame({
    "fitted": fitted_values,
    "residual": residuals,
    "studentized_residual": studentized_residuals,
    "leverage": leverage,
    "cooks_distance": cooks_distance
}, index=model.model.data.row_labels)

# ============================================================
# 5. Diagnostic plots
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 11)
)

# Residuals versus fitted values
sns.residplot(
    x=fitted_values,
    y=residuals,
    lowess=True,
    scatter_kws={"alpha": 0.35},
    line_kws={"color": "red"},
    ax=axes[0, 0]
)

axes[0, 0].axhline(0, color="black", linestyle="--")
axes[0, 0].set_title("Residuals vs Fitted Values")
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")


# Normal Q-Q plot
sm.qqplot(
    residuals,
    line="45",
    fit=True,
    ax=axes[0, 1]
)

axes[0, 1].set_title("Normal Q-Q Plot")


# Scale-location plot
sqrt_abs_studentized = np.sqrt(
    np.abs(studentized_residuals)
)

axes[1, 0].scatter(
    fitted_values,
    sqrt_abs_studentized,
    alpha=0.35
)

sns.regplot(
    x=fitted_values,
    y=sqrt_abs_studentized,
    scatter=False,
    lowess=True,
    line_kws={"color": "red"},
    ax=axes[1, 0]
)

axes[1, 0].set_title("Scale-Location Plot")
axes[1, 0].set_xlabel("Fitted Values")
axes[1, 0].set_ylabel(
    "Square Root of Absolute Studentized Residual"
)


# Residuals versus leverage
axes[1, 1].scatter(
    leverage,
    studentized_residuals,
    alpha=0.35
)

axes[1, 1].axhline(0, color="black", linestyle="--")

high_leverage_cutoff = 2 * p / n
axes[1, 1].axvline(
    high_leverage_cutoff,
    color="red",
    linestyle="--",
    label=f"Leverage cutoff = {high_leverage_cutoff:.4f}"
)

axes[1, 1].set_title("Residuals vs Leverage")
axes[1, 1].set_xlabel("Leverage")
axes[1, 1].set_ylabel("Studentized Residuals")
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# 6. Influential observations
# ============================================================

print("\n========== INFLUENTIAL OBSERVATIONS ==========")

cook_cutoff = 4 / n

print("Cook's distance cutoff:", cook_cutoff)
print("High-leverage cutoff:", high_leverage_cutoff)

influential_observations = diagnostics_df[
    (diagnostics_df["cooks_distance"] > cook_cutoff)
    | (diagnostics_df["leverage"] > high_leverage_cutoff)
    | (np.abs(diagnostics_df["studentized_residual"]) > 3)
].sort_values(
    "studentized_residual",
    ascending=False,
    key=lambda x: x.abs()
)

print(
    influential_observations.head(25).to_string()
)

print()
print()
pd.set_option("display.float_format", "{:.2f}".format)
rows = df.loc[influential_observations.head(25).index, ["Year", "Age", "WorkExp", "DevType", "ConvertedCompYearly"]]

print(rows.to_string())
# ============================================================
# 7. Heteroskedasticity test
# ============================================================

print("\n========== BREUSCH-PAGAN TEST ==========")

bp_results = het_breuschpagan(
    residuals,
    X
)

bp_labels = [
    "LM statistic",
    "LM p-value",
    "F statistic",
    "F p-value"
]

print(dict(zip(bp_labels, bp_results)))


# ============================================================
# 8. Durbin-Watson statistic
# ============================================================

print("\n========== DURBIN-WATSON ==========")
print(durbin_watson(residuals))


# ============================================================
# 9. Save diagnostic tables if desired
# ============================================================

vif_df.to_csv("individual_vif_results.csv", index=False)
term_vif_df.to_csv("term_vif_results.csv", index=False)
diagnostics_df.to_csv("observation_diagnostics.csv")

from statsmodels.stats.multitest import multipletests
import pandas as pd

# Exclude the intercept unless you specifically want to test it
p_values = model.pvalues.drop("Intercept")

reject, p_corrected, _, _ = multipletests(
    p_values,
    method="fdr_bh"
)

results = pd.DataFrame({
    "term": p_values.index,
    "original_p": p_values.values,
    "adjusted_p": p_corrected,
    "significant_after_fdr": reject
})

print(results.to_string(index=False))

wald_table = model.wald_test_terms(
    skip_single=False,
    scalar=True
).table.copy()

print(wald_table.columns)  # optional: inspect column names

joint_results = wald_table.reset_index()
joint_results = joint_results.rename(columns={
    "index": "term",
    "pvalue": "original_p"
})

p_values = joint_results["original_p"].astype(float)

reject, adjusted_p, _, _ = multipletests(
    p_values,
    method="fdr_bh"
)

joint_results["adjusted_p"] = adjusted_p
joint_results["significant_after_fdr"] = reject

print(joint_results.to_string(index=False))
