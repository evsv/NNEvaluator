import pandas as pd
import numpy as np
import random

def float_to_str(df, obs, col):
    """Randomly select given no. of elements in a column & convert float to str

    :param df: <Pandas Dataframe> dataframe to made dirty
    :param trigger: <Int> number of elements to be made dirty
    :param trigger: <Int> the Index of the column to be made dirty
    :return: <Pandas Dataframe> return the modified dataframe
    """

    # randomly select given number of rows and change values in given column
    for idx in df.sample(n=obs).index:
        df.set_value(idx, col, str(df.loc[idx][col]) + '', takeable=True)
    return df

def float_to_int(df, obs, col):
    """Randomly select given no. of elements in a column & convert float to int

    :param df: <Pandas Dataframe> dataframe to made dirty
    :param trigger: <Int> number of elements to be made dirty
    :param trigger: <Int> the Index of the column to be made dirty
    :return: <Pandas Dataframe> return the modified dataframe
    """
    # randomly select given number of rows and change values in given column
    for idx in df.sample(n=obs).index:
        df.set_value(idx, col, int(df.loc[idx][col]), takeable=True)
    return df

def make_dirty(df, frac_rows, frac_cols):
    """Make an input dataset dirty based on size & input dirty fractions

    :param df: <Pandas Dataframe> dataframe to made dirty
    :param frac_rows: <float> fraction of observations to be made dirty
    :param frac_cols: <float> fraction of columns to be made dirty
    :return: <Pandas Dataframe> return the modified dataframe
    """
    N = len(df)
    iterations = int(len(df.columns)*frac_cols)
    frac_dirty = int(N*frac_rows)

    # make randomly selected elements dirty
    for i in range(iterations):
        
        # choose column & number of observations to make dirty
        obs = random.randint(int(frac_dirty/2), frac_dirty)
        col = random.randint(0,len(df.columns))
        
        flag = random.randint(0, 1)
        if flag == 0:
            df = float_to_str(df, obs, col)
        else:
            df = float_to_int(df, obs, col)

    return df