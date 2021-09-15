import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import acquire

## -------------------------- Main Function -------------------------- ##

def wrangle_zillow_2():
    """ Acquires, cleans, splits, and scales zillow data for full model, returns:
        df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test
        Version 2 includes dummy columns for FIPS code """
    df = acquire.acquire_zillow()
    df = clean_data_2(df)
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_isolate_data(df)
    X_train_exp, X_train, X_validate, X_test = scale_data_2(X_train, X_validate, X_test)
    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

def wrangle_zillow_1():
    """ Acquires, cleans, splits, and scales zillow data for full model, returns:
        df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test
        Version 1 does not cast FIPS code to dummy values """
    df = acquire.acquire_zillow()
    df = clean_data_1(df)
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_isolate_data(df)
    X_train_exp, X_train, X_validate, X_test = scale_data_1(X_train, X_validate, X_test)
    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test


def wrangle_zillow_MVP():
    """ Acquires, cleans, splits, and scales zillow data for MVP, returns:
    df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test """
    df = acquire.acquire_zillow()
    df = clean_data_MVP(df)
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_isolate_data(df)
    X_train_exp, X_train, X_validate, X_test = scale_data(X_train, X_validate, X_test)
    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

## ---------------------- Individual Functions ----------------------- ##

def clean_data_2(df):
    """ Cleans the columns used for the full effect model, returns clean df """
    # Fixes columns
    df['ID'] = df['ID'].astype('O')
    df['LocalityCode'] = df['LocalityCode'].astype('int').astype('str')
    # Adds new 'Age' column
    df['Age'] = 2017 - df['YearBuilt']
    df['TaxRate'] = round((df['Taxes'] / df['Worth']) * 100, 3)
    # Maps county names to FIPS column
    map1 = {'6037':'Los Angeles County, CA', '6059':'Orange County, CA', '6111':'Ventura County, CA'}
    df['LocalityCode'] = df.LocalityCode.map(map1)
    # Limits columns
    cols = ['ID', 'LocalityCode', 'DateSold', 'Worth', 'TaxRate', 'Baths', 'Beds', 'LotSize',
            'FinishedSize', 'Age']
    df = df[cols]
    # Drops nulls and duplicates
    df = df.dropna().drop_duplicates()
    # Removes outliers
    df = remove_outliers(df, k=1.5, col_list=cols[3:])
    # Adds dummy columns for localities
    df = pd.get_dummies(data=df, columns=['LocalityCode'], drop_first=True)

    return df

def clean_data_1(df):
    """ Cleans the columns used for the full effect model, returns clean df """
    # Fixes columns
    df['ID'] = df['ID'].astype('O')
    df['LocalityCode'] = df['LocalityCode'].astype('int').astype('str')
    # Adds new 'Age' column
    df['Age'] = 2017 - df['YearBuilt']
    df['TaxRate'] = round((df['Taxes'] / df['Worth']) * 100, 3)
    # Maps county names to FIPS column
    map1 = {'6037':'Los Angeles County, CA', '6059':'Orange County, CA', '6111':'Ventura County, CA'}
    df['LocalityCode'] = df.LocalityCode.map(map1)
    # Limits columns
    cols = ['ID', 'LocalityCode', 'DateSold', 'Worth', 'TaxRate', 'Baths', 'Beds', 'LotSize',
            'FinishedSize', 'Age']
    df = df[cols]
    # Drops nulls and duplicates
    df = df.dropna().drop_duplicates()
    # Removes outliers
    df = remove_outliers(df, k=1.5, col_list=cols[3:])

    return df

def clean_data_MVP(df):
    """ Cleans the columns used for the minimum viable product """
    cols = ['Worth','FinishedSize','Beds','Baths']
    df = df[cols]
    df = df.dropna().drop_duplicates()
    df = remove_outliers(df, k=1.5, col_list=cols)
    return df

def split_isolate_data(df):
    """ Splits data and isolates target into X_train, y_train, etc., then returns to:
        X_train, y_train, X_validate, y_validate, X_test, y_test """
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)
    X_train, y_train = train.drop(columns='Worth'), train.Worth
    X_validate, y_validate = validate.drop(columns='Worth'), validate.Worth
    X_test, y_test = test.drop(columns='Worth'), test.Worth
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def scale_data_2(X_train, X_validate, X_test):
    """ Standard-Scales the zillow numeric columns I selected, returns to:
        X_train_exp, X_train, X_validate, X_test """
    # Set numeric columns to scale
    col_list = ['Baths', 'Beds', 'LotSize', 'FinishedSize', 'Age',
                'LocalityCode_Orange County, CA', 'LocalityCode_Ventura County, CA']
    # Build and fit scaler using numeric columns
    scaler = StandardScaler().fit(X_train[col_list])
    # Builds new dataframe with col_scaled values in new columns
    X_train_exp = X_train.copy()
    col_list_scaled = []
    for col in col_list:
        col_list_scaled.append(col + "_scaled")
    X_train_exp[col_list_scaled] = scaler.transform(X_train[col_list])
    # Reassigns X splits to array of scaled values
    X_train = scaler.transform(X_train[col_list])
    X_validate = scaler.transform(X_validate[col_list])
    X_test = scaler.transform(X_test[col_list])
    # Sends everything back
    return X_train_exp, X_train, X_validate, X_test

def scale_data_1(X_train, X_validate, X_test):
    """ Standard-Scales the zillow numeric columns I selected, returns to:
        X_train_exp, X_train, X_validate, X_test """
    # Set numeric columns to scale
    col_list = ['Baths', 'Beds', 'LotSize', 'FinishedSize', 'Age']
    # Build and fit scaler using numeric columns
    scaler = StandardScaler().fit(X_train[col_list])
    # Builds new dataframe with col_scaled values in new columns
    X_train_exp = X_train.copy()
    col_list_scaled = []
    for col in col_list:
        col_list_scaled.append(col + "_scaled")
    X_train_exp[col_list_scaled] = scaler.transform(X_train[col_list])
    # Reassigns X splits to array of scaled values
    X_train = scaler.transform(X_train[col_list])
    X_validate = scaler.transform(X_validate[col_list])
    X_test = scaler.transform(X_test[col_list])
    # Sends everything back
    return X_train_exp, X_train, X_validate, X_test

def remove_outliers(df, col_list, k=1.5):
    """ 
        Uses Interquartile Rule to remove outliers from dataframe;
        Requires dataframe and list of columns to remove outliers from,
        Accepts an optional k-value (default set to k=1.5),
        Returns full dataframe without outliers in specified columns.
    """
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # Get quartiles
        iqr = q3 - q1   # Calculate interquartile range
        upper_bound = q3 + k * iqr   # Get upper bound
        lower_bound = q1 - k * iqr   # Get lower bound
        # Create mask
        mask = (df[col] > lower_bound) & (df[col] < upper_bound)
        df = df[mask] # Apply mask
    return df

## ------------------------- Plot Functions -------------------------- ##

def initial_plots(df):
    """ Plots univariate distributions of dataframe's numeric columns for
        values before and after applying the interquartile rule for outliers """
    for col in df.columns:
        """ Plots """
        # Raw plots
        sns.histplot(df, x=col)
        plt.title(col)
        plt.show()
        
        # Interquartile Rule
        if (df[col].dtype == 'float64') or (df[col].dtype == 'int64'):
            q1, q3 = df[col].quantile([.25, .75])  # get quartiles
            k = 1.5
            iqr = q3 - q1   # calculate interquartile range
            upper_bound = q3 + k * iqr   # get upper bound
            lower_bound = q1 - k * iqr   # get lower bound
            mask = (df[col] > lower_bound) & (df[col] < upper_bound)
            temp = df[mask]
            sns.histplot(temp, x=col)
            plt.title(col + '_no_outliers')
            plt.show()