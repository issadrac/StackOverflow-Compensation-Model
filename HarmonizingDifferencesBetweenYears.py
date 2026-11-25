import pandas as pd

column_list = [
    "Year",
    "Age",
    "Country",
    "EdLevel",
    "OrgSize",
    "YearsCodePro",
    "WorkExp",
    "Industry",
    "DevType",
    "ConvertedCompYearly",
    "ConvertedCompYearlyInflation",
    "LanguageHaveWorkedWith",
    "LanguageWantToWorkWith",
    "AISelect",
    "AISent",
    "AIAcc",
    "AIComplex"
]

def standardize_age(age_column):
    numeric_age = pd.to_numeric(age_column, errors="coerce")

    age_ranges = pd.cut(numeric_age, bins=[float("-inf"), 17, 24, 34, 44, 54, 64, float("inf")],
        labels=[
            "Under 18 years old",
            "18-24 years old",
            "25-34 years old",
            "35-44 years old",
            "45-54 years old",
            "55-64 years old",
            "65 years or older",
        ],
    )

    # Use the converted range for numeric ages. Keep existing text ranges.
    return age_ranges.astype("string").fillna(age_column.astype("string"))

def standardize_edlevel(edlevel_column):
    values = edlevel_column.astype("string").str.strip()

    result = pd.Series(pd.NA, index=values.index, dtype="string")

    result[values.str.contains("Bachelor", case=False, na=False)] = "Bachelor's degree"

    result[values.str.contains("Master", case=False, na=False)] = "Master's degree"

    result[values.str.contains("Associate", case=False, na=False)] = "Associate degree"

    result[values.str.contains("Professional degree", case=False, na=False)] = "Professional degree"

    result[values.str.contains("Doctoral", case=False, na=False)] = "Professional degree"

    result[values.str.contains("Some college", case=False, na=False)] = "Some college"

    result[values.str.contains("Secondary school", case=False, na=False)] = "Secondary school"

    result[values.str.contains("Primary/elementary", case=False, na=False)] = "Primary/elementary school"

    result[values.str.contains("never completed", case=False, na=False)] = "No formal education"

    # Nonmissing values not matched above
    unmatched = values.notna() & result.isna()
    result[unmatched] = "Other"

    return result

def standardize_convertedcompyearly(convertedcompyearly_column, year_column, base_year=2025):
    annual_cpi = {
        2019: 255.657,
        2020: 258.811,
        2021: 270.970,
        2022: 292.655,
        2023: 304.702,
        2024: 313.689,
        2025: 321.943,
        2026: 336.94,
    }

    salary = pd.to_numeric(
        convertedcompyearly_column,
        errors="coerce"
    )

    years = pd.to_numeric(
        year_column,
        errors="coerce"
    )

    cpi_current_year = years.map(annual_cpi)
    cpi_base_year = annual_cpi[base_year]

    return salary * (cpi_base_year / cpi_current_year)

def prepare_year(filename, year, column_sources=None):
    filename = "StackOverflowData/" + filename
    df = pd.read_csv(filename, low_memory=False)
    column_sources = column_sources or {}

    result = pd.DataFrame(index=df.index)
    result["Year"] = year

    for output_column in column_list[1:]:
        source_column = column_sources.get(output_column, output_column)

        if source_column in df.columns:
            result[output_column] = df[source_column]
        else:
            result[output_column] = pd.NA

    # Fixing Age
    result["Age"] = standardize_age(result["Age"])
    # Fixing EdLevel
    result["EdLevel"] = standardize_edlevel(result["EdLevel"])
    # Fixing Country Difference (US & UK)
    country_mapping = {
        "United States":"United States of America",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    }
    result["Country"] = result["Country"].replace(country_mapping)
    # Converting Compensation for Inflation
    result["ConvertedCompYearlyInflation"] = (standardize_convertedcompyearly(result["ConvertedCompYearly"], result["Year"]))
   
    # Standardizing DevType
    def normalize_multiselect(value, mapping):
        if pd.isna(value):
            return value

        return ";".join(mapping.get(option.strip(), option.strip()) for option in str(value).split(";"))
    role_mapping = {
        "Educator or instructor": "Educator",
        "UX, Research Ops or UI design professional": "Designer",
        "UX, research, or UI design professional": "Designer",
        "Developer advocate or developer relations professional": "Developer Advocate",
        "Founder or entrepreneur": "Founder, technology or otherwise",
        "Senior executive (C-suite, VP, etc.)":"Senior Executive (C-Suite, VP, etc.)",
        "Senior executive/VP":"Senior Executive (C-Suite, VP, etc.)",
        "Applied scientist": "Scientist",
        "DevOps engineer or professional": "DevOps specialist",
        "Cybersecurity or InfoSec professional": "Security professional",
        "Site reliability engineer": "Engineer, site reliability",
        "Database administrator or engineer": "Database administrator",
        "Data scientist": "Data scientist or machine learning specialist",
        "Marketing/Sales professional": "Marketing or sales professional",
        "Engineer, data":"Data engineer",
        "Hardware Engineer":"Hardware engineer",
        "Developer, AI apps or physical AI": "Developer, AI",
        "Other (please specify):": "Other"
    }
    result["DevType"] = result["DevType"].apply(normalize_multiselect, mapping=role_mapping)

    industry_mapping = {
        # Rewording
        'Banking/Financial Services':'Banking/financial services',
        'Financial Services':'Banking/financial services',
        'Software Development':'Software development',
        'Media & Advertising Services':'Media or advertising services',
        'Advertising Services':'Media or advertising services',
        'Retail and Consumer Services':'Retail, e-commerce, or consumer services',
        #'Transportation, or Supply Chain':'Manufacturing, Transportation, or Supply Chain',
        'Transportation or supply chain':'Transportation, or Supply Chain',
        #'Manufacturing':'Manufacturing, Transportation, or Supply Chain',
        
        # Combining
        'Education':'Higher Education',
        'Utilities, internet, or telecommunications':'Internet, Telecomm or Information Services',
        'Broadcasting or publishing': 'Media or advertising services',
        'IT services, consulting, systems integration, or managed services':'Computer Systems Design and Services',
        'Oil & Gas':'Energy',

        # Other
        'Other:':'Other',
        'Other, please specify':'Other',
        'Agriculture, forestry, fishing, and hunting':'Other',
        'Engineering services or applied R&D': 'Other',
        'Hospitality or travel': 'Other',
        "Scientific research": "Other",
        'Non-profit or social services': 'Other',
        'Life sciences or biotechnology':'Other',
        'Mining or construction':'Other',
    }

    result["Industry"] = result["Industry"].replace(industry_mapping)

    aisentiment_mapping = {
        "Not sure":"Unsure",
        "Neutral":"Indifferent",
        "Somewhat favorable":"Favorable",
        "Somewhat unfavorable":"Unfavorable",
    }
    result["AISent"] = result["AISent"].replace(aisentiment_mapping)

    orgsize_mapping = {
        "2 to 9 employees":"Less than 20 employees",
        "10 to 19 employees": "Less than 20 employees"
    }
    result["OrgSize"] = result["OrgSize"].replace(orgsize_mapping)

    return result


def input_2019():
    return prepare_year("2019.csv", 2019,
        {
            "ConvertedCompYearly": "ConvertedComp",
            "LanguageHaveWorkedWith": "LanguageWorkedWith",
            "LanguageWantToWorkWith": "LanguageDesireNextYear",
        }
    )

def input_2020():
    return prepare_year("2020.csv", 2020,
        {
            "ConvertedCompYearly": "ConvertedComp",
            "LanguageHaveWorkedWith": "LanguageWorkedWith",
            "LanguageWantToWorkWith": "LanguageDesireNextYear",
        },
    )

def input_2021():
    return prepare_year("2021.csv", 2021)

def input_2022():
    return prepare_year("2022.csv", 2022)

def input_2023():
    return prepare_year("2023.csv", 2023, {"AIAcc": "AIBen"})

def input_2024():
    return prepare_year("2024.csv", 2024)

def input_2025():
    return prepare_year("2025.csv", 2025, {"YearsCodePro":"WorkExp"})

def input_2026():
    return prepare_year("2026.csv", 2026, 
        {
            "YearsCodePro":"WorkExp",
            "LanguageHaveWorkedWith": "LanguagesHaveWorkedWith",
            "LanguageWantToWorkWith": "LanguagesWantToWorkWith",
            "DevType":"Role",
            "EdLevel":"Education",
            "AISent":"AISentiment",
        },
    )


main_data_frame = pd.concat(
    [
        input_2019(),
        input_2020(),
        input_2021(),
        input_2022(),
        input_2023(),
        input_2024(),
        input_2025(),
        input_2026(),
    ],
    ignore_index=True,
)
main_data_frame["ID"] = main_data_frame.index
#print(main_data_frame.fillna("NA"))
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

# Next 25
#     ID  Year                Age                   OrgSize  WorkExp                                        DevType  ConvertedCompYearly  ConvertedCompYearlyInflation
# 470082  2025    35-44 years old  10,000 or more employees    11.00  Data scientist or machine learning specialist           2346000.00                    2346000.00
# 433171  2024    25-34 years old    Less than 20 employees    10.00                          Developer, full-stack           1200000.00                    1231575.22
# 469999  2025    25-34 years old  10,000 or more employees    11.00                            Developer, back-end           1950000.00                    1950000.00
# 516673  2026    25-34 years old  10,000 or more employees    12.00                            Developer, back-end           1950000.00                    1863206.65
# 536944  2026    25-34 years old  10,000 or more employees    12.00                            Developer, back-end           1900000.00                    1815432.12
# 435564  2024    25-34 years old  10,000 or more employees    11.00                            Developer, back-end           1900000.00                    1949994.10
# 492048  2025    35-44 years old      500 to 999 employees    10.00                       Data or business analyst            878500.00                     878500.00
# 526087  2026    35-44 years old  10,000 or more employees    21.00                              Developer, mobile           2300000.00                    2197628.36
# 359834  2023    35-44 years old      100 to 499 employees    18.00                            Developer, back-end           1800000.00                    1901849.68
# 491190  2025    18-24 years old      100 to 499 employees     3.00           Senior Executive (C-Suite, VP, etc.)            950000.00                     950000.00
# 478447  2025    25-34 years old      500 to 999 employees     8.00                          Developer, full-stack           1000000.00                    1000000.00
# 408133  2024    45-54 years old  10,000 or more employees    25.00    Developer, embedded applications or devices           1500000.00                    1539469.03
# 319444  2023    35-44 years old  1,000 to 4,999 employees    17.00                          Developer, full-stack           1300000.00                    1373558.10
# 361765  2023    25-34 years old  10,000 or more employees     7.00                                          Other           1250000.00                    1320728.94
# 487182  2025    35-44 years old  1,000 to 4,999 employees    17.00                          Developer, full-stack           1279775.00                    1279775.00
# 342059  2023  65 years or older    Less than 20 employees    50.00  Developer, desktop or enterprise applications           1000000.00                    1056583.15
# 383527  2023    35-44 years old    Less than 20 employees    16.00  Developer, desktop or enterprise applications            750000.00                     792437.37
# 467937  2025    45-54 years old  10,000 or more employees    23.00                            Developer, back-end           1500000.00                    1500000.00
# 405958  2024    18-24 years old      100 to 499 employees     2.00                                        Student            225000.00                     230920.35
# 337645  2023    35-44 years old      100 to 499 employees    23.00                            Engineering manager           1500000.00                    1584874.73
# 485847  2025    45-54 years old      100 to 499 employees    18.00                            Developer, back-end             25000.00                      25000.00
# 542462  2026    18-24 years old  5,000 to 9,999 employees     1.00                          Developer, full-stack             13221.00                      12632.54
# 318878  2023    25-34 years old      100 to 499 employees     6.00                          Developer, full-stack            850000.00                     898095.68
# 520478  2026    45-54 years old    Less than 20 employees    20.00                              Developer, mobile             20000.00                      19109.81
# 475977  2025    25-34 years old  10,000 or more employees     7.00                                          Other             21000.00                      21000.00

bogus_index_list = [474697, 522691, 310821, 536853, 514999,
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
          516785, 529191,
          # Next 25
          470082,
          433171,
          469999,
          516673,
          536944,
          435564,
          492048,
          526087,
          359834,
          491190,
          478447,
          408133,
          319444,
          361765,
          487182,
          342059,
          383527,
          467937,
          405958,
          337645,
          485847,
          542462,
          318878,
          520478,
          475977]

main_data_frame = main_data_frame[~main_data_frame["ID"].isin(bogus_index_list)]

# Specific to Compensation Model

years = range(2023, 2027)
main_data_frame = main_data_frame[main_data_frame["Year"].isin(years)]
main_data_frame = main_data_frame[main_data_frame["Country"] == "United States of America"]
# 0 values breaks the model because of np.log
# Confirmed that it was most likely mistakes given how few results there were espeically since for most roles, 
# the exact same number is the same for 500, 1000, 3000, 5000, 10000
main_data_frame = main_data_frame[main_data_frame["ConvertedCompYearly"] > 10000]


# Convert AISelect to 0/1
main_data_frame.dropna(subset=["AISelect"], inplace=True)
main_data_frame["AISelect"] = (~(main_data_frame["AISelect"].str.contains("No,") | (main_data_frame["AISelect"].str.contains("I don") & ~main_data_frame["AISelect"].str.contains(";")))).astype("int8")


main_data_frame.to_csv("combined_2023_2026.csv", index=False, na_rep="NA")