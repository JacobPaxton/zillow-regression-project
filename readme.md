This repository contains my work for the Zillow Regression project.

Project Overview: Analyze Zillow data from May to August 2017 using 'Single Family Residential' observations from the 'zillow' dataset from the Codeup database. Build a regression model to predict home values. Present a slide deck containing work and findings.

Plan: 
1. Acquire the correct data subset from the 'zillow' dataset
    * Notebook: Filter to May-August 2017, Single Family Residential
        * SQL: JOIN predictions_2017 USING(parcelid) WHERE residence-type = 261
    * Notebook: Store to .csv file
    * **ACQUIRE.py**: 
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

3. Check univariate distributions
    * Notebook: Use square-footage, bedroom-count, bathroom-count, and tax-value for initial exploration
    * *After MVP*: check new features for viability
    * **PREP.py**: 
        * Add initial-plots function to loop through and plot features
    * **Present**: 
        * First four distributions
        * *After MVP*: Any additional distributions included/excluded

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
    * Notebook: Create baseline model
        * Calculate mean and median of target (tax-value)
        * Assign mean and median to columns in y_train and y_validate
        * Calculate RMSE for both train and validate
            * mean_squared_error(actuals, baseline) ** 0.5
        * Keep the lower-error baseline (of mean and median)
    * Notebook: Create models for different regression algorithms
        * Loop through one algorithm's hyperparameters, save to list
        * Loop through next algorithm, and next... using same
    * Notebook: Loop lists of models through model-performance function
        * Extend the 'model_name' to include hyperparameter
        * Add to same dataframe for easy column-wise analysis
    * Notebook: "Choose" best-performing model
    * **Present**: 
        * model-performance function
        * baseline performance
        * model performance
        * model selected

6. Revisit Step #3, add more features, then adjust Step #4 and #5 with features that have the potential to improve your models

7. 