import numpy as np
import pandas as pd
from copy import deepcopy
from keras.optimizers import SGD, RMSprop

class EvalClass:

    def __init__(self, model, optimiser_grid = ["sgd"], 
                 loss_grid = "mse", lr_grid = [0.01]):

        self._model = model
        self._optimiser_grid = optimiser_grid
        self._loss_grid = loss_grid
        self._fit_verb_flag = True

    @property
    def optimiser_grid(self):
        return self._optimiser_grid

    @optimiser_grid.setter
    def optimiser_grid(self, og):
        self._optimiser_grid = og

    @property
    def loss_grid(self):
        return self._loss_grid

    @loss_grid.setter
    def loss_grid(self, lg):
        self._loss_grid = lg

    @property
    def fit_verb_flag(self):
        return self._fit_verb_flag

    @fit_verb_flag.setter
    def fit_verb_flag(self, verb):
        self._fit_verb_flag = verb

    @property
    def optim_perf_list(self):
        return self._optim_perf_list

    def model_compile(self, model, optimiser, loss, metrics, learning_rate = None):

        if optimiser == "sgd":
            model.compile(optimizer = SGD(lr = learning_rate),
                                loss = loss, metrics = metrics)
        elif optimiser == "rmsprop":
            model.compile(optimizer = RMSprop(lr = learning_rate),
                          loss = loss, metrics = metrics)
        else:
            model.compile(optimizer = optimiser, loss = loss, 
                                metrics = metrics)

    def eval_optimiser(self, x_train, y_train, valid_perc = 0.2,
                       epochs = 50, batch_size = 32,
                       num_of_iterations = 1, learning_rate = 0.001,
                       loss = "mse", metrics = ["mse", "mae", "acc"]):

        self._optim_perf_list = {k: [] for k in self._optimiser_grid}
        optim_model_list = {k: [] for k in self._optimiser_grid}

        # ITERATING THROUGH EACH OPTIMISER TO EVALUATE MODEL PERFORMANCE
        for optim in self.optimiser_grid:
            
            # INITIALISING MODEL
            model = deepcopy(self._model)
            orig_weights = model.get_weights()

            # COMPILING MODEL
            self.model_compile(model = model, optimiser = optim, 
                               loss = loss, metrics = metrics, 
                               learning_rate = learning_rate)

            # iterating user specified number of times to compile distribution
            # of model metrics
            for i in range(0, num_of_iterations):
                print(model.get_weights())
                history = model.fit(x_train, y_train, 
                                    validation_split = valid_perc,
                                    epochs = epochs, batch_size = batch_size,
                                    verbose = self.fit_verb_flag)
                
                self._optim_perf_list[optim].append(history)
                optim_model_list[optim].append(model)

                # RESETTING MODEL WEIGHTS
                model.set_weights(orig_weights)

        return optim_model_list


                