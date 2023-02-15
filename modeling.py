import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing
np.random.seed(99)
from sklearn.metrics import mean_squared_error
from math import sqrt 
from sklearn.model_selection import train_test_split
import modeling
from sklearn.preprocessing import MinMaxScaler

def evaluate(target_var, val, yhat_df):
    '''
    This function will take the actual values of the target_var from val, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(val[target_var], yhat_df[target_var])), 0)
    return rmse

def plot_and_eval(target_var, train, val_t, yhat_df):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, val, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1, color='#377eb8')
    plt.plot(val_t[target_var], label='val', linewidth=1, color='#ff7f00')
    plt.plot(yhat_df[target_var], label='yhat', linewidth=2, color='#a65628')
    plt.legend()
    plt.title(target_var)
    rmse = evaluate(target_var, val_t, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()
    
def append_eval_df(model_type, target_var, val_t, yhat_df, eval_df):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, val_t, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)

def make_baseline_predictions(cpue_predictions, val_t):
    yhat_df = pd.DataFrame({'cpue': [cpue_predictions]},
                          index=val_t.index)
    return yhat_df


def final_plot(target_var, train, val_t, test_t, yhat_df):
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], color='#377eb8', label='train')
    plt.plot(val_t[target_var], color='#ff7f00', label='val')
    plt.plot(test_t[target_var], color='#4daf4a',label='test_t')
    plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
    plt.legend()
    plt.title(target_var)
    plt.show()
    
    
def train_val_test(df):
    '''
    this function splits up the data into sections for training,
    validating, and testing
    models
    '''
    seed = 99
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5, random_state = seed)
    
    return train, val, test

def split_time(df):
    train_size = int(round(df.shape[0] * 0.5))
    validate_size = int(round(df.shape[0] * 0.3))
    test_size = int(round(df.shape[0] * 0.2))
    
    validate_end_index = train_size + validate_size
    
    train = df[:train_size]
    validate = df[train_size:validate_end_index]
    test_t = df[validate_end_index:]
    
    return train, validate, test_t

def dummy_fab(train, val, test):
    train_m = pd.get_dummies(train, columns = ['sex','year'], drop_first = [True, True])
    val_m = pd.get_dummies(val, columns = ['sex','year'], drop_first = [True, True])
    test_m = pd.get_dummies(test, columns = ['sex','year'], drop_first = [True, True])
    
    return train_m, val_m, test_m
    
    
def scale_4_model(train, val, test):
    to_scale = ['latitude', 'longitude','bottom_depth','surface_temperature','bottom_temperature']
    mm_scaler = MinMaxScaler()
    for col in to_scale:
        mm_scaler.fit(train[[col]])
        train[col] = mm_scaler.transform(train[[col]])
        val[col] = mm_scaler.transform(val[[col]])
        test[col] = mm_scaler.transform(test[[col]])
    x_train = train.drop(columns = 'cpue')
    y_train = train['cpue']
    x_val = val.drop(columns = 'cpue')
    y_val = val['cpue']
    x_test = test.drop(columns = 'cpue')
    y_test = test['cpue']
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def modeling_prep(train, val, test):
    train, val, test = dummy_fab(train, val, test)
    x_train, y_train, x_val, y_val, x_test, y_test =modeling.scale_4_model(train, val, test)
    return x_train, y_train, x_val, y_val, x_test, y_test
    
    
def time_split(df):
    train_len = int(0.6 *len(df))
    val_test_split = int(0.8 * len(df))
    train_t = df.iloc[:train_len]
    val_t = df.iloc[train_len: val_test_split]
    test_t = df.iloc[val_test_split:]
    return train_t, val_t, test_t