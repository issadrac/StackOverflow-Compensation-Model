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
    # Fixing United States Difference
    result["Country"] = result["Country"].replace("United States", "United States of America")
    # Converting Compensation for Inflation
    result["ConvertedCompYearly"] = (standardize_convertedcompyearly(result["ConvertedCompYearly"], result["Year"]))
   
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
#print(main_data_frame.fillna("NA"))
main_data_frame.to_csv("combined_2019_2026.csv", index=False, na_rep="NA")
