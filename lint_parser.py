import pandas as pd
import numpy as np

# Data Linter to coersion function mapper
COERCE = {'DateTimeAsStringDetector': 'string_to_date',\
          'TokenizableStringDetector': 'to_categorical',\
          'ZipCodeAsNumberDetector': 'number_to_zipcode',\
          'NumberAsStringDetector': 'string_to_number',\
          'NonNormalNumericFeatureDetector': 'rescale_col',\
          'UniqueValueCountsDetector': 'carry_on',\
          'IntAsFloatDetector': 'float_to_int',\
          'UncommonSignDetector': 'carry_on',\
          'DuplicateExampleDetector': 'carry_on',\
          'EmptyExampleDetector': 'carry_on',\
          'UncommonListLengthDetector': 'carry_on',\
          'TailedDistributionDetector': 'remove_tail',\
          'CircularDomainDetector': 'carry_on'}

# Given a Data Linter triggered, identify the columns to coerce
def cols_to_coerce(lines, trigger, result_dict, coerce_type):
    """Identify the columns to coerce for a given Data Linter triggered

    :param lines: <List> the lines of the text file from the linter
    :param trigger: <String> the name of the Data Linter triggered
    :param result_dict: <Dictionary> column to coersion type mapping
    :param coerce_type: <Enum> the type of coercion to affect the column
    :return: <Pandas Dataframe> return the modified dictionary
    """

    # identify the output file section corresponding to the trigger
    i = lines.index(trigger)
    j = lines[i+2:].index('='*80)

    # parse text lines to pick out columns triggering the data linter
    result_dict[coerce_type] = []
    k = j - 1
    while lines[i+2:][k][0] == '*':
        line = lines[i+2:][k]
        col = line.split()[1]

        result_dict[coerce_type].append(col)
        k -= 1

    return result_dict

def parse_output(lint_output_path):
    """Parse linter output to coerce dataset as per linter recommendation.

    :param lint_output_path: <String> the path where the data linter output is stored
    :return: <Dictionary> return mapping of columns for each coersion required
    """
    result_dict = {}

    # read in the desired Data Linter Output
    lines = [line.rstrip('\n') for line in open(lint_output_path)]

    # identify the Data Linters triggered on the dataset
    linters_triggered = []
    i = 1
    while lines[i] != '':
        linters_triggered.append(lines[i][2:])
        i += 1

    # For each Data Linter triggered, identify the columns to coerce
    for trigger in linters_triggered:

        coerce_type = COERCE[trigger]
        if coerce_type != 'carry_on':    #ignore selected linter types
            result_dict = cols_to_coerce(lines, trigger, result_dict, coerce_type)

    return result_dict
