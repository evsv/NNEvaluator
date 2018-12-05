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

def make_dirty(df):
    """Make an input dataset dirty based on size

    :param df: <Pandas Dataframe> dataframe to made dirty
    :return: <Pandas Dataframe> return the modified dataframe
    """
    N = len(df)
    iterations = max(10, int(len(df.columns)/3))

    # make randomly selected elements dirty
    for i in range(iterations):
        obs = max(100, random.randint(1, int(N*0.01)))
        col = random.randint(0,len(df.columns))
        flag = random.randint(0, 1)

        if flag == 0:
            df = float_to_str(df, obs, col)
        else:
            df = float_to_int(df, obs, col)

    return df