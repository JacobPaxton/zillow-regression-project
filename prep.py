import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import acquire

## -------------------------- Main Function -------------------------- ##

def wrangle_zillow_MVP():
    df = acquire.acquire_zillow()
    df = clean_data_MVP(df)
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_isolate_data(df)
    X_train_exp, X_train, X_validate, X_test = scale_data(X_train, X_validate, X_test)
    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

## ---------------------- Individual Functions ----------------------- ##

def clean_data_MVP(df):
    cols = ['Worth','FinishedSize','Beds','Baths']
    df = df[cols]
    df = df.dropna().drop_duplicates()
    df = remove_outliers(df, k=1.5, col_list=cols)
    return df

def split_isolate_data(df):
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)
    X_train, y_train = train.drop(columns='Worth'), train.Worth
    X_validate, y_validate = validate.drop(columns='Worth'), validate.Worth
    X_test, y_test = test.drop(columns='Worth'), test.Worth
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def scale_data(X_train, X_validate, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_exp = X_train.copy()
    col_list = []
    for col in X_train.columns:
        col_list.append(col + "_scaled")
    X_train_exp[col_list] = scaler.transform(X_train)
    X_train = scaler.transform(X_train)
    X_validate = scaler.transform(X_validate)
    X_test = scaler.transform(X_test)
    return X_train_exp, X_train, X_validate, X_test

def remove_outliers(df, col_list, k=1.5):
    """ 
        Uses Interquartile Rule to remove outliers from dataframe,
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