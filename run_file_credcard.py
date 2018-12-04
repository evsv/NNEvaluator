import pandas
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras import layers
from  EvaluatorClass import EvalClass

# READING IN DATA
dataZip = zipfile.ZipFile("creditcard.csv.zip")
df = pd.read_csv(dataZip.open("creditcard.csv"))
df_covar = df.iloc[:, 0:30]
df_dep = df.iloc[:, 30].values

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
                                     epochs = 10, metrics = ["accuracy"], num_of_iterations = 2)

optim_mod_list

eval.study_eval_results(optim_mod_list)