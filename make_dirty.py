import pandas as pd
import numpy as np
import random

def make_dirty(df, frac_rows, frac_cols):
    """Make an input dataset dirty based on size & input dirty fractions

    :param df: <Pandas Dataframe> dataframe to made dirty
    :param frac_rows: <float> fraction of observations to be made dirty
    :param frac_cols: <float> fraction of columns to be made dirty
    :return: <Pandas Dataframe> return the modified dataframe
    """
    N = len(df)
    dirty_rows = int(N*frac_rows)
    dirty_cols = int(len(df.columns)*frac_cols)
    
    col_list = np.random.choice(df.columns, dirty_cols, replace=False)
    for col in col_list:

        # generate a list of randomly selected indices
        rand_indices = np.random.choice(df.index, dirty_rows, replace=False)
        
        flag = random.randint(0, 1)
        if flag == 0:
            df.loc[rand_indices, col] = df.loc[rand_indices, col].apply(lambda x: str(x) + '')

        else:
            df.loc[rand_indices, col] = df.loc[rand_indices, col].apply(lambda x: int(x*1.0))

    return df