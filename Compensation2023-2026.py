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
df["AISelect"].dropna(inplace=True)
df["AISelect"] = (~(df["AISelect"].str.contains("No,") | (df["AISelect"].str.contains("I don") & ~df["AISelect"].str.contains(";")))).astype("int8")

# Finding bogus results [Everything is user-inputted]
#         Year                Age                                             OrgSize  WorkExp                                        DevType  ConvertedCompYearly  ConvertedCompYearlyInflation
# 474697  2025    25-34 years old                            5,000 to 9,999 employees     2.00                            Academic researcher           9000000.00                    9000000.00
# 522691  2026    35-44 years old                            5,000 to 9,999 employees    16.00                            Developer, back-end          30000000.00                   28664717.75
# 310821  2023    25-34 years old                            10,000 or more employees     2.00  Data scientist or machine learning specialist           9000000.00                    9509248.38
# 536853  2026    55-64 years old                              Less than 20 employees    50.00  Developer, desktop or enterprise applications           6000000.00                    5732943.55
# 514999  2026  65 years or older                                  20 to 99 employees    54.00           Senior Executive (C-Suite, VP, etc.)             11000.00                      10510.40
# 499647  2025    35-44 years old                              Less than 20 employees    15.00                          Developer, full-stack           2000000.00                    2000000.00
# 492730  2025    45-54 years old  Just me - I am a freelancer, sole proprietor, etc.    20.00                          Security professional           3900000.00                    3900000.00
# 486648  2025    35-44 years old                                100 to 499 employees    19.00                            Developer, back-end             13000.00                      13000.00
# 379231  2023    35-44 years old                                        I don’t know    20.00                            Engineering manager             11900.00                      12573.34
# 518627  2026    35-44 years old                                  20 to 99 employees    11.00                                       Designer           2000000.00                    1910981.18
# 518564  2026    35-44 years old                            10,000 or more employees    15.00                            Developer, back-end           3500000.00                    3344217.07
# 542612  2026    35-44 years old                            1,000 to 4,999 employees    17.00                          Developer, full-stack             11800.00                      11274.79
# 518363  2026    35-44 years old                                  20 to 99 employees    25.00                          Developer, full-stack             12000.00                      11465.89
# 518733  2026    35-44 years old                                  20 to 99 employees    13.00                              DevOps specialist             10800.00                      10319.30
# 420052  2024    45-54 years old                                  20 to 99 employees    24.00                          Developer, full-stack             10000.00                      10263.13
# 366993  2023    35-44 years old                            10,000 or more employees    10.00                           Developer, front-end           2130000.00                    2250522.12
# 466983  2025    35-44 years old                                100 to 499 employees    10.00                          Developer, full-stack             11000.00                      11000.00
# 482072  2025    25-34 years old                              Less than 20 employees     9.00           Senior Executive (C-Suite, VP, etc.)             15000.00                      15000.00
# 521733  2026    25-34 years old                                500 to 999 employees     5.00    Developer, embedded applications or devices             10700.00                      10223.75
# 420107  2024    55-64 years old                                100 to 499 employees    30.00                          Developer, full-stack             11000.00                      11289.44
# 342951  2023    55-64 years old                              Less than 20 employees    38.00                                Project manager             11000.00                      11622.41
# 543718  2026    55-64 years old                                        I don’t know    21.00                          Developer, full-stack           1000000.00                     955490.59
# 493466  2025    55-64 years old                              Less than 20 employees    15.00                          Developer, full-stack           1000000.00                    1000000.00
# 477737  2025    25-34 years old                            1,000 to 4,999 employees     6.00                          Developer, full-stack             12000.00                      12000.00
# 326717  2023    45-54 years old                            10,000 or more employees    35.00                                          Other           2400000.00                    2535799.57
# 503654  2025    35-44 years old                            5,000 to 9,999 employees     5.00  Developer, desktop or enterprise applications             12000.00                      12000.00
# 540165  2026    45-54 years old                            5,000 to 9,999 employees    25.00                              DevOps specialist             18000.00                      17198.83
# 323760  2023    45-54 years old                              Less than 20 employees    36.00                                Project manager             12000.00                      12679.00
# 527930  2026    45-54 years old  Just me - I am a freelancer, sole proprietor, etc.     7.00                                  Data engineer           1200000.00                    1146588.71
# 429718  2024    25-34 years old                            10,000 or more employees     6.00    Developer, embedded applications or devices             18500.00                      18986.78
# 520368  2026    18-24 years old                                100 to 499 employees     5.00                            Developer, back-end             12000.00                      11465.89
# 512849  2025    18-24 years old                            10,000 or more employees     5.00                          Developer, full-stack           1000000.00                    1000000.00
# 364111  2023    45-54 years old                                100 to 499 employees    30.00                          Developer, full-stack             19000.00                      20075.08
# 417067  2024    25-34 years old                              Less than 20 employees     5.00                            Developer, back-end             15000.00                      15394.69
# 480473  2025    45-54 years old                                  20 to 99 employees    31.00                          Developer, full-stack             18900.00                      18900.00
# 492337  2025    18-24 years old                            1,000 to 4,999 employees     7.00                           Developer, front-end             11470.00                      11470.00
# 520075  2026    18-24 years old                            10,000 or more employees     2.00  Data scientist or machine learning specialist             13000.00                      12421.38
# 429633  2024    35-44 years old                                  20 to 99 employees    10.00                          Developer, full-stack             17000.00                      17447.32
# 496429  2025    35-44 years old                            5,000 to 9,999 employees    10.00                          Developer, full-stack             18000.00                      18000.00
# 426567  2024  65 years or older                            5,000 to 9,999 employees    43.00                                      Scientist             22000.00                      22578.88
# 512472  2025    25-34 years old                              Less than 20 employees    12.00                          Developer, full-stack             13000.00                      13000.00
# 418336  2024    35-44 years old                                  20 to 99 employees    20.00                            Engineering manager             26000.00                      26684.13
# 321993  2023    25-34 years old                                  20 to 99 employees     5.00                                          Other             14400.00                      15214.80
# 496442  2025    35-44 years old                                        I don’t know    19.00                                          Other             18000.00                      18000.00
# 437763  2024    35-44 years old                              Less than 20 employees     8.00                              Developer, mobile             20000.00                      20526.25
# 521525  2026    45-54 years old                              Less than 20 employees    27.00                          Developer, full-stack             20000.00                      19109.81
# 517599  2026    35-44 years old                            10,000 or more employees    15.00                          Developer, full-stack             27300.00                      26084.89

# Hobbyists, part-time, or retired folk
# 544149  2026    55-64 years old  Just me - I am a freelancer, sole proprietor, etc.    42.00                          Developer, full-stack             12000.00                      11465.89
# 515068  2026    35-44 years old  Just me - I am a freelancer, sole proprietor, etc.    16.00                          Developer, full-stack             12400.00                      11848.08
# 310371  2023    55-64 years old  Just me - I am a freelancer, sole proprietor, etc.    32.00                  Cloud infrastructure engineer             15400.00                      16271.38
# 470934  2025    45-54 years old  Just me - I am a freelancer, sole proprietor, etc.    30.00                            Developer, back-end             15000.00                      15000.00
# 529420  2026    55-64 years old  Just me - I am a freelancer, sole proprietor, etc.    40.00                              Developer, mobile             19600.00                      18727.62
# 516785  2026    25-34 years old  Just me - I am a freelancer, sole proprietor, etc.     3.00                          Developer, full-stack             12000.00                      11465.89
# 529191  2026    25-34 years old                              Less than 20 employees     7.00                           Developer, front-end             16000.00                      15287.85

df = df.drop(
   index=[474697, 522691, 310821, 536853, 514999,
          499647, 492730, 486648, 379231, 518627, 
          518564, 542612, 518363, 518733, 420052, 
          366993, 466983, 482072, 521733, 420107, 
          342951, 543718, 493466, 477737, 326717, 
          503654, 540165, 323760, 527930, 429718, 
          520368, 512849, 364111, 417067, 480473, 
          492337, 520075, 429633, 496429, 426567, 
          512472, 418336, 321993, 496442, 437763, 
          521525, 517599,
          # Hobbyists, part-time, or retired folk
          544149, 515068, 310371, 470934, 529420,
          516785, 529191],
   errors="ignore").copy()


# Make it so that only valid selections are added
import CompareYears as cy
import statsmodels.api as sm
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
model = sm.OLS.from_formula(formula, data=df).fit(cov_type="HC3")

print(model.summary())
model.save("Final_Model.pickle")
# This tests the model by training on all years but the final year and testing the model based on that final year
# Ex: Observed Years 2020-2026
#   Training Data: 2020-2025
#   Test Data = 2026
cy.GetTestResults(df, years, formula)
# Testing original model with a multitude of tests
# Basic diagnostics
# VIF (group and ungrouped)
# Residual Diagnostic with plots
# Influential Observations
# Heteroskedasticity test
# Durbin-Watson statistic
# Wald z-tests (Individual and joint)
# Multiple testing correction
cy.AI_Tests(df, model, print_influential_instance_results=True, original_mean=original_mean)