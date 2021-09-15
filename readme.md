# Project Overview
Analyze Zillow data from May to August 2017 using 'Single Family Residential' observations from the 'zillow' dataset from the Codeup database. Build a regression model to predict home values. Present a slide deck containing work and findings.

# Goals
1. Deliver residence locality information
2. Deliver distribution chart for tax rates by locality
3. Build regression models for MVP restriction that predicts property value better than the baseline
4. Identify the drivers of property value
5. Improve regression models with additional features
6. Present work and findings to an audience familiar with data science
7. Deliver my work in the form of a clean Jupyter Notebook

# Hypotheses
1. There is a linear relationship between a home's value and the number of bedrooms it has.
    * Hypothesis is true with 95% confidence.
2. The value of a home with two bedrooms is not statistically different from a home with two bathrooms.
    * Hypothesis did not pass the 95% confidence threshold.
3. The value of homes in Orange County is higher than the value of homes in all three counties combined
    * Before removing outliers, this hypothesis is true with 95% confidence.
    * After removing outliers, this hypothesis did not pass the 95% confidence threshold.
4. The values of Age before the interquartile rule is applied is statistically significantly different from the values of Age after the interquartile rule.
    * Hypothesis did not pass the 95% confidence threshold.

# Plan 
1. Acquire the correct data subset from the 'zillow' dataset
    * Notebook: Filter to May-August 2017, Single Family Residential
        * SQL: JOIN predictions_2017 USING(parcelid) WHERE residence-type = 261
    * Notebook: Store to .csv file
    * **ACQUIRE.py**: 
        * Add get_db_url
        * Add acquire-store function
    * **Present**:
        * Acquisition steps and data subsecting
2. Create a distribution for residence locality (state, county) against tax rate
    * Notebook: Explore data for locality information
        * **Deliverable**: List of states and counties where properties are located
    * Notebook: Calculate tax rate using property value and tax amount
    * Notebook: Distibution x-axis: tax_rate, y-axis: value_count, hue/color: locality
        * **Deliverable**: This distribution
    * **Present**: 
        * All localities for dataset's residences
        * Each locality's tax rate range
        * Each peak in the tax rate distributions
3. Conduct hypothesis testing and check univariate distributions
    * Notebook: Initial exploration
        * Use square-footage, bedroom-count, bathroom-count, and tax-value for initial exploration
        * *After MVP*: Run initial exploration on new features
    * Notebook: Hypothesis testing
        * Create initial hypotheses for MVP features, push to readme
        * Run statistical tests
            * At least two statistical tests along with visualizations documenting hypotheses and takeaways
        * Convey results to readme and Jupyter notebook
        * *After MVP*: Create hypotheses for new features 
        * *After MVP*: Push new features through hypothesis testing to check viability
        * *After MVP*: Convey results to readme and Jupyter notebook
    * **PREP.py**: 
        * Add initial-plots function to loop through and plot features
    * **Present**: 
        * First four distributions
        * *After MVP*: Any additional distributions included/excluded
        * Initial hypotheses
        * Statistical tests
        * Results
4. Prepare using Minimum-Viable-Product (MVP) specification restriction
    * Notebook: Drop all columns except square-footage, bedroom-count, bathroom-count, tax-value
    * Notebook: Drop all nulls in above columns
    * Notebook: Check for outliers using box-and-whisker plot
    * Notebook: Eliminate outliers if needed using Inter-Quartile Rule
    * Notebook: Rename columns to something more readable
    * Notebook: Split data into train, validate, and test
    * Notebook: Isolate target variable 'tax-value' into y_train from X_train
        * Do the same for X_validate and X_test
    * Notebook: Scale data
        * Create and fit scaler using X_train
        * Create X_train_exp using scaler transform of X_train while retaining original values
        * Scale X_train, drop unscaled columns
        * Scale X_validate, drop unscaled columns
        * Scale X_test, drop unscaled columns
    * *After MVP*: Run through above steps as needed with any additional features
    * **PREP.py**: 
        * Add plot-data function to make various plots for a dataframe when called
        * Add clean-data function to limit dataset features, drop nulls, eliminate outliers, rename columns
            * *After MPV*: Revise clean-data with new features
        * Add split-data function for train/validate/test *and* target isolation
        * Add scale-data function
        * Add wrangle-data function to run acquire-store, clean-data, split-data, and scale-data functions, then return all dataframes
    * **Present**:
        * Overview of wrangling, mentioning feature limitation to MVP, additional features, nulls, outliers, feature renaming, split, target isolation, scaler creation, scaler application, returned dataframes
5. Create models for MVP restriction
    * Notebook: Cast y_train and y_validate as dataframes
    * Notebook: Create model-performance function
        * Takes in actuals_series, predictions_series, 'model_name', df_to_append_to
        * Calculates RMSE
        * Calculates r^2 score
        * Appends dataframe with new row for model_name, RMSE_validate, r^2_score
        * Returns dataframe
    * Notebook: Create plot-residuals function
    * Notebook: Create baseline model
        * Calculate mean and median of target (tax-value)
        * Assign mean and median to columns in y_train and y_validate
        * Calculate RMSE for both train and validate
            * mean_squared_error(actuals, baseline) ** 0.5
        * Keep the lower-error baseline (of mean and median)
        * Call plot-residuals
    * Notebook: Create models for different regression algorithms
        * Loop through one algorithm's hyperparameters, save to list
        * Loop through next algorithm, and next... using same
    * Notebook: Loop lists of models through model-performance function
        * Extend the 'model_name' to include hyperparameter
        * Add to same dataframe for easy column-wise analysis
        * Call plot-residuals
    * Notebook: "Choose" best-performing model
        * Plot y by yhat
    * *After MVP*: Add features, use k-best or RFE to determine which features to include
    * *After MVP*: Loop model-performance using new feature set and suitable names
    * *After MVP*: "Choose" best-performing model
    * **Present**: 
        * model-performance function
        * baseline performance
        * MVP model performance
        * After-MVP model performance
        * model selected
6. Revisit Step #3, #4, and #5 with more features than the MVP restriction
    * Complete these steps:
        * Run at least 1 t-test and 1 correlation test (but as many as you need!)
        * Visualize all combinations of variables in some way(s).
        * What independent variables are correlated with the dependent?
        * Which independent variables are correlated with other independent variables?
    * Run all *After MVP* steps
7. Push work and findings to a slide deck
8. Practice/script the presentation
9. Present!

# Data Dictionary
| Feature           | Datatype                | Definition   |
|:------------------|:------------------------|:-------------|
| ID                | 19631 non-null: object  | Unique identifier for the property            |
| LocalityCode      | 19631 non-null: object  | The three counties the dataset's coverage     |
| DateSold          | 19631 non-null: object  | Date the property was sold                    |
| Worth             | 19631 non-null: float64 | Tax-assessed value of the property, dollars   |
| TaxRate           | 19631 non-null: float64 | Tax rate for the property, percentage         |
| Baths             | 19631 non-null: float64 | Number of bathrooms on the property           |
| Beds              | 19631 non-null: float64 | Number of bedrooms on the property            |
| LotSize           | 19631 non-null: float64 | Size of the lot, sqft                         |
| FinishedSize      | 19631 non-null: float64 | Size of the property's finished area, sqft    |
| Age               | 19631 non-null: float64 | Age of the home, years                        |

# Feature Selection notes
1. Worth: Chose taxvaluedollarcnt instead of landtaxvaluedollarcnt because the value of the property is what we're looking at
2. Baths: Chose bathroomcnt because it had more values available than the other bathroom columns
2. Beds: Chose bedroomcnt because it had more values available than the other bedroom columns
3. FinishedSize: Chose calculatedfinishedsquarefeet because it had more values available than the other finishedsquarefeet columns
4. Age: Calculated age from yearbuilt column to have a feature with values starting from zero

# Instructions to Replicate My Work
1. Clone this repository
2. Add your own env.py file for server credentials
3. Execute the cells in final_notebook.ipynb