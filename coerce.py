import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

STRING_TO_NUMBER = 1
STRING_TO_DATE = 2
FLOAT_TO_INT = 3
NUMBER_TO_ENUM = 4
TO_CATEGORICAL = 5
NUMBER_TO_ZIPCODE = 6

def coerce_column(df, column_name, lint_code):
    """Coerce columns according to linter recommendation.

    :param df: <Pandas Dataframe>
    :param column_name: <String> the name of the column to coerce
    :param lint_code: <Integer> the code of the lint recommendation
    :return: <Pandas Dataframe> return the modified dataframe
    """
    if lint_code == STRING_TO_NUMBER:
        return string_to_number(df, column_name)
    elif lint_code == STRING_TO_DATE:
        return string_to_date(df, column_name)
    elif lint_code == FLOAT_TO_INT:
        return float_to_int(df, column_name)
    elif lint_code == TO_CATEGORICAL:
        return to_categorical(df, column_name)
    elif lint_code == NUMBER_TO_ZIPCODE:
        return number_to_zipcode(df, column_name)

def to_categorical(df, column_name):
    """Encode a string column to a one-hot categorical column.

    :param df: <Pandas Dataframe>
    :param column_name: <String> the name of the column to coerce
    :return: <Pandas Dataframe> return the modified dataframe
    """
    df = df.join(pd.get_dummies(df[column_name]))
    df = df.drop(column_name, axis=1)
    return df

def string_to_number(df, column_name):
    """Coerce a string column to a numeric column.

    :param df: <Pandas Dataframe>
    :param column_name: <String> the name of the column to coerce
    :return: <Pandas Dataframe> return the modified dataframe
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def string_to_date(df, column_name):
    """Coerce a string column to a datetime column.

    :param df: <Pandas Dataframe>
    :param column_name: <String> the name of the column to coerce
    :return: <Pandas Dataframe> return the modified dataframe
    """
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    return df

def float_to_int(df, column_name):
    """Coerce a numeric/float column to an integer column.

    :param df: <Pandas Dataframe>
    :param column_name: <String> the name of the column to coerce
    :return: <Pandas Dataframe> return the modified dataframe
    """
    df[column_name] = df[column_name].fillna(0.0).astype(int)
    return df

def number_to_zipcode(df, column_name):
    """Coerce a numeric/float column to a string zipcode column.

    :param df: <Pandas Dataframe>
    :param column_name: <String> the name of the column to coerce
    :return: <Pandas Dataframe> return the modified dataframe
    """
    df[column_name] = df[column_name].apply(lambda x: str(x).strip()[0:5] if len(str(x).strip()) > 5 else str(x).strip())
    return df

# TODO
def convert_types_cc(df):
    #eg. if column "A1" needed translation to categorical
    df = coerce_column(df, "A1", TO_CATEGORICAL)
