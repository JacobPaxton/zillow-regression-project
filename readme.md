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

# Key Findings
1.

# Initial Hypotheses
1.

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
        * Add plot-data dunction to make various plots for a dataframe when called
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

# Notes to self
- "You will want to make sure you are using the best fields to represent square feet of home, number of bedrooms, and number of bathrooms. "Best" meaning the most accurate and available information. Here you will need to do some data investigation in the database and use your domain expertise to make some judgement calls."
- "Brainstorming ideas and form hypotheses related to how variables might impact or relate to each other, both within independent variables and between the independent variables and dependent variable."
- "Document any ideas for new features you may have while first looking at the existing variables and the project goals ahead of you."
- "Add a data dictionary in your notebook at this point that defines all the fields used in your model and your analysis and answers the question, "Why did you use the fields you used?". e.g. "Why did you use bedroom_field1 over bedroom_field2?", not, "Why did you use number of bedrooms?""