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
        """
        Evaluator class interface to call the Keras Sequential Model Class'
        ".compile" method, based on the selected optimiser.

        :param model: The Model or EValModel object to be compiled
        :type model: Keras Sequential Model or EvalModel object
        :param optimiser: Optimiser to be used in the model fitting process
                          Currently takes values:
                            1. sgd
                            2. rmsprop
        :type optimiser: str
        :param loss: Loss function to be used in the model fitting process.
                     Takes loss values defined within the Keras API.
        :type loss: str
        :param metrics: Metrics to be tracked over the fitting process.
                        Takes values defined within the Keras API.
        :type metrics: str (or) list of str
        :param learning_rate: learning rate to be used by the optimiser.
                              Defaults to None. The optimisers which currently
                              use the learning rate parameter are:
                                1. SGD
                                2. RMSPROP
        :param learning_rate: num, optional
        """

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
                       loss = "mse", metrics = ["mse", "mae", "acc"], callbacks=None):
        """
        Evaluation function to iterate over a sequence of optimiser options,
        fit the model to track time to train and other metrics, and return list
        of fitted models.

        To better estimate the performance of different choices, models can
        be repeated a user a specified number of times to construct a
        distribution of performance measures.

        :param x_train: Covariate training data
        :type x_train: np array
        :param y_train: Dependent training data
        :type y_train: np array
        :param valid_perc: percentage of training data to be used as a
                           validation sample, defaults to 0.2
        :param valid_perc: float, optional
        :param epochs: Number of epochs to train the model for, defaults to 50
        :param epochs: int, optional
        :param batch_size: Batch size of each bath used in training,
                           defaults to 32
        :param batch_size: int, optional
        :param num_of_iterations: Number of times model fitting process should
                                  take place, defaults to 1
        :param num_of_iterations: int, optional
        :param learning_rate: Learning rate to be used during the model
                              compilation process, defaults to 0.001
                              Optimisers which currently support the learning
                              rate parameter are:
                                1. SGD
                                2. RMSProp
        :param learning_rate: float, optional
        :param loss: Loss metric to be used in the training process, defaults
                     to "mse"
        :param loss: str, optional
        :param metrics: Metrics to be tracked over the fitting process.
                        Takes values defined within the Keras API,
                        defaults to ["mse", "mae", "acc"]
        :param metrics: list of str, optional
        :param callback_list: Callbacks to be tracked through the training 
                              process
        :param callback_list: list of Keras Callbacks, optional
        :return: [description]
        :rtype: [type]
        """


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
                                    verbose = self.fit_verb_flag, callbacks=callbacks)

                self._optim_perf_list[optim].append(history)
                optim_model_list[optim].append(model)

                # RESETTING MODEL WEIGHTS
                model.set_weights(orig_weights)

        return optim_model_list

    def study_eval_results(self, model_dict):

        # CHECKING THE KIND OF HISTORY OBJECT PROVIDED
        history_keys = model_dict.keys()
        if set(history_keys) == set(self.optimiser_grid):
            grid_type = "AN OPTIMISER"
        elif set(history_keys) == set(self.optimiser_grid):
            grid_type = "A LOSS"

        key_str = ", ".join(history_keys)
        print("THE PROVIDED HISTORY OBJECT CONTAINS MODEL EVALUATIONS OVER "
              + grid_type + " GRID AND THE PARAMETERS ARE: " + key_str)

        # PRINTING INFORMATION ON EVALUATION RUN
        print("DEBUG: history_keys")
        print(history_keys)
        history_keys = list(history_keys)
        metric_list = list(model_dict[history_keys[0]][0].history.history.keys())
        metric_str = ", ".join(metric_list)
        print("THE METRICS TRACKED FOR THIS EVALUATION ARE: " + metric_str)

        # ITERATING ACROSS EACH PARAMETER IN THE PARAMETER GRID
        for key in history_keys:

            print("\n\nLOOKING AT PERFORMANCE CORRESPONDING TO PARAMETER: "
                  + key)
            history_obj_list = [model_obj.history
                                    for model_obj in model_dict[key]]

            # Iterating across each metric being tracked
            for metric in metric_list:
                print(metric)
                metric_array = [obj.history[metric]
                                    for obj in history_obj_list]

                print(metric_array)
