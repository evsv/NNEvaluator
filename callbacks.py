import keras
from time import time
import numpy as np

class ConvergeTime(keras.callbacks.Callback):

    def __init__(self, min_epoch_run, perc_change_thresh, num_of_priors):
        
        self.min_epoch_run = min_epoch_run
        self.perc_change_thresh = perc_change_thresh
        if num_of_priors <= min_epoch_run:
            self.num_of_priors = num_of_priors
        elif num_of_priors <= 1:
            num_of_priors = 2
        else:
            self.num_of_priors = min_epoch_run

    def on_train_begin(self, logs={}):
        
        self.loss_list = []
        self.time_till_epoch_list = []
        self.epoch_num = 0
        self.start_time = time()
        self.if_converged = False
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        
        # GETTING EPOCH STATS
        self.loss_list.append(logs.get("val_loss"))
        self.time_till_epoch_list.append(time() - self.start_time)
        
        # START CHECKING FOR CONVERGENCE
        loss_improvement_list = [i-j for i, j in zip(self.loss_list[:-1], self.loss_list[1:])]
        if self.epoch_num >= self.min_epoch_run:
            if not self.if_converged:
                # obtaining loss value for user specified number of priod periods
                loss_value_window = self.loss_list[-1*self.num_of_priors:]

                # finding the improvement between subsequent epochs
                loss_improvement_list = [i-j for i, j in zip(loss_value_window[:-1], 
                                                            loss_value_window[1:])]
                
                # finding percentage decrease in loss value between subsequent
                # epochs
                loss_value_denominator = loss_value_window[0:(self.num_of_priors - 1)]
                lil_np_array = np.array(loss_improvement_list)
                lvd_np_array = np.array(loss_value_denominator)
                perc_loss_decr = lil_np_array/lvd_np_array

                # checking for overfitting condition, i.e, condition where 
                # validation loss starts increasing
                if len(np.where(perc_loss_decr < 0)[0]) != 0:

                    epoch_overfit_index = min(np.where(perc_loss_decr < 0)[0])
                    
                    # setting converged flag to true and recording convergence conditions
                    self.if_converged = True
                    self.converged_loss = loss_value_window[epoch_overfit_index]
                    self.converged_time = self.time_till_epoch_list[self.num_of_priors + epoch_overfit_index - 1]

                    return

                # checking for convergence condition, i.e, condition where all
                # epochs defined by num_of_priors parameter observe a percentage
                # decrease less than the specified threshold
                if sum((perc_loss_decr <= self.perc_change_thresh)) == (self.num_of_priors - 1):

                    self.if_converged = True
                    self.converged_loss = self.loss_list[self.epoch_num]
                    self.converged_time = self.time_till_epoch_list[self.epoch_num]

                    return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class RunTime(keras.callbacks.Callback):

    def on_train_begin(self, logs = {}):
        self.start_time = time()
        self.run_time = 0

    def on_train_end(self, logs = {}):
        run_time = time() - self.start_time
        self.run_time = run_time