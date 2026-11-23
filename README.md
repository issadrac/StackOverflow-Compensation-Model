# StackOverflow-Compensation-Model
#### How to run:
- Download all the results.csv and rename to fit the program.
- https://github.com/StackExchange/Survey/tree/main/packages/archive
- Install pandas and statsmodels
- HarmonizingDifferencesBetweenYears.py creates the combined_2019_2026.csv file
- Compensation2023-2026.py creates the model using the data from combined_2019_2026.csv with the statsmodels library.
- CompareYears has a lot of helper functions that is used in Compensation2023-2026.py
- The output from Compensation2023-2026.py is in model_coef_and_tests
