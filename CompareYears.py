import pandas as pd
import numpy as np
import re 

def find_multiselect_options(df, column):
    """
    Finds options that appear inside ;X; in a row and if that happens then the column is considered multiselect
    """
    values = df[column].dropna().astype(str).str.strip()

    # Find options appearing in the middle of a semicolon-separated value
    middle_options = (
        values
        .str.extractall(r";\s*([^;]+?)\s*;")[0]
        .str.strip()
        .dropna()
        .unique()
    )
    
    if len(middle_options) > 0:
        return df[column].dropna().str.split(";").explode().str.strip().unique()
    else:
        return []


def compareColumn(df, years, column, getOutput=False, PRINT=True):
    allOfThem = []
    for year in years:
        yearly_data = df[df["Year"]==year]
        multiselectOptions = find_multiselect_options(yearly_data, column)
        if PRINT:
            print(year)
        if len(multiselectOptions) > 0:
            if PRINT:
                print(multiselectOptions)
            if getOutput:
                allOfThem.append(multiselectOptions)
        else:
            if PRINT:
                print(yearly_data[column].unique())
            if getOutput:
                allOfThem.append(yearly_data[column].unique())
        if PRINT:
            print()
    if PRINT:
        print()
    if getOutput:
        return allOfThem

def FindWhatsNotIn(df, years, column, printIn=True, printNotIn=True):
    selection_all_years = compareColumn(df, years, column, True, False)
    set_selections = [{value for value in selection if pd.notna(value)} for selection in selection_all_years]
    
    current = 0
    for year in years:
        addingAllButOneSet = set()
        for i, set_year in enumerate(set_selections):
            if i != current:
                addingAllButOneSet.update(set_year)
            else:
                if printIn:
                    print(f"Variables in {year}:")
                    print(sorted(set_year))
        notIn = addingAllButOneSet - set_selections[current]
        if printNotIn:
            print(f"Everything not included in {year} for column: {column}")
            print(sorted(notIn))
        if printNotIn or printIn:
            print()
        current += 1
   
def FindWhatsInAll(df, years, column, PRINT=False):
    selection_all_years = compareColumn(df, years, column, True, False)

    set_selections = [{value for value in selection if pd.notna(value)} for selection in selection_all_years]
    if not set_selections:
        return set()

    combiningSet = set_selections[0]
    for set_year in set_selections[1:]:
        combiningSet.intersection_update(set_year)
    if PRINT:
        print("All selections shared in years:", years)
        print(sorted(combiningSet))

    return combiningSet

def GetCounts(df, years, column, string, NOT=False):
    counts = {}
    for year in years:
        yearly_data = df[df["Year"] == year]
        matches = yearly_data[column].astype("string").str.contains(string, na=False, regex=False)
        if NOT:
            matches = ~matches
        counts[year] = int(matches.sum())

    return counts

def CountsPerYear(df, years, column, MULTI=False, PRINT=True):
    results = []

    for year in years:
        yearly_data = df[df["Year"] == year].copy()

        valid_mask = yearly_data[column].notna()

        if MULTI:
            valid_mask &= yearly_data[column].astype(str).str.strip().ne("")

            values = (
                yearly_data[column]
                .fillna("N/A")
                .astype(str)
                .str.split(";")
                .explode()
                .str.strip()
                .replace("", "N/A")
            )
            counts = values.value_counts(dropna=False)
        else:
            counts = yearly_data[column].value_counts(dropna=False)

        year_counts = counts.rename_axis(column).reset_index(name="Count")
        year_counts.insert(0, "Year", year)

        total_respondents = len(yearly_data)
        total_non_na = valid_mask.sum()

        year_counts["Percentage"] = (
            year_counts["Count"] / total_respondents * 100
            if total_respondents > 0
            else 0
        )

        year_counts["Percentage_No_NA"] = (
            year_counts["Count"] / total_non_na * 100
            if total_non_na > 0
            else 0
        )

        # N/A should not receive a percentage in the no-NA calculation
        year_counts.loc[
            year_counts[column].astype(str).eq("N/A"),
            "Percentage_No_NA"
        ] = pd.NA

        results.append(year_counts)

        if PRINT:
            print("Year:", year)
            print(year_counts)
            print()

    return (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(
            columns=[
                "Year",
                column,
                "Count",
                "Percentage",
                "Percentage_No_NA",
            ]
        )
    )
    
#years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
#FindWhatsInAll(years, "DevType")


def GetTestResults(df, years, formula, PRINT=True):
    import statsmodels.formula.api as smf
    
    # Split the data
    train_df = df[df["Year"] < years[-1]-years[0]].copy()
    test_df = df[df["Year"] == years[-1]-years[0]].copy()
    
    # Fit your original model on training data only
    model_test = smf.ols(
        formula=formula,
        data=train_df
    ).fit(cov_type="HC3")
    
    # Predict log compensation on unseen data
    y_actual = np.log(test_df["ConvertedCompYearlyInflation"])
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
        np.log(train_df["ConvertedCompYearlyInflation"]).mean(),
        len(test_df)
    )
    
    y_actual = np.log(test_df["ConvertedCompYearlyInflation"])
    
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

def AI_Tests(df, model, print_influential_instance_results=False, original_mean = 0):
    # AI TESTS ----------------------------------
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
    if print_influential_instance_results:
        df["WorkExp"] = df["WorkExp"] + original_mean
        df["Year"] = df["Year"] + 2023
        rows = df.loc[influential_observations.head(25).index, ["Year", "Age", "OrgSize", "WorkExp", "DevType", "ConvertedCompYearly", "ConvertedCompYearlyInflation"]]
        df["WorkExp"] = df["WorkExp"] - original_mean
        df["Year"] = df["Year"] - 2023
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
    
    vif_df.to_csv("Created/output/individual_vif_results.csv", index=False)
    term_vif_df.to_csv("Created/output/term_vif_results.csv", index=False)
    diagnostics_df.to_csv("Created/output/observation_diagnostics.csv")
    
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