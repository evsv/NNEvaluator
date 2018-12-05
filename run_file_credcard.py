import pandas
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras import layers
from  EvaluatorClass import EvalClass
from callbacks import ConvergeTime
from make_dirty import make_dirty

import coerce

LINT_DATA = False
# READING IN DATA
dataZip = zipfile.ZipFile("creditcard.csv.zip")
df = pd.read_csv(dataZip.open("creditcard.csv"))
df = make_dirty(df, .1, .8)

# READING IN LINTS
if LINT_DATA:
    df = coerce.preprocess(df, "lint_outputs/cc_results.txt")
    df.tail()

df_covar = df.iloc[:, 0:30]
df_dep = df.iloc[:, 30].values

t0 = time.time()
# NORMALISING DATA
scaler = preprocessing.StandardScaler().fit(df_covar)
df_covar_scaled = scaler.transform(df_covar)

# TEST-TRAIN SPLIT
len(df.columns.values)
covar = df_covar_scaled[:, 0:30]
dep = df_dep
print(covar)
covar_train, covar_test, dep_train, dep_test = train_test_split(covar, dep, test_size=0.33, random_state=42)
type(dep_test)
# BUILDING THE MODEL
test_model = Sequential()
test_model.add(layers.Dense(units=512, activation='linear', input_dim = 30))
test_model.add(layers.Dense(units=128, activation='linear'))
test_model.add(layers.Dense(units=32, activation='linear'))
test_model.add(layers.Dropout(0.5))
test_model.add(layers.Dense(units=1, activation='sigmoid'))

eval = EvalClass(model = test_model, optimiser_grid = ["rmsprop"])
optim_mod_list = eval.eval_optimiser(x_train = covar_train, y_train = dep_train,
                                     loss = "binary_crossentropy", learning_rate=2e-5,
                                     epochs = 15, metrics = ["accuracy"], num_of_iterations = 1, callbacks=[ConvergeTime(5, .0005, 3, stop_on_converge=True)])

t1 = time.time()
print("Took {} seconds".format(t1-t0))
