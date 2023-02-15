import pandas as pd
from scipy import stats
from pydataset import data
import numpy as np
import env
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def train_val_test(df):
    '''
    this function splits up the data into sections for training,
    validating, and testing
    models
    '''
    seed = 99
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    validate, test = train_test_split(val_test, train_size = 0.5, random_state = seed)
    return train, validate, test

def get_snow_crabs():
    df = pd.read_csv('mfsnowcrab.csv')
    df = df[df['cpue'] < 2000000]
    df['year'] = pd.to_datetime(df['year'], format = '%Y')
    df.drop(columns = ['id', 'name','haul'], inplace = True)
    df_time = df[['year','cpue']]
    df_time = df_time.set_index('year')
    df_time = df_time.sort_index()
    df_time = df_time.resample('y')['cpue'].mean()
    df = pd.DataFrame(df)
    df_time = pd.DataFrame(df_time)
    return df , df_time
