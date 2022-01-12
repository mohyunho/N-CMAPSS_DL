#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual

"""
import pandas as pd
from abc import abstractmethod

# from input_creator import input_gen
from utils.elm_network import network_fit



def release_list(lst):
   del lst[:]
   del lst

class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    '''
    TODO: Consider hyperparameters of ELM instead of the number of neurons in hidden layers of MLPs.
    Class for EA Task
    '''
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, constant, batch, model_path, device, obj):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.constant = constant
        self.batch = batch
        self.model_path = model_path
        self.device = device
        self.obj = obj

    def get_n_parameters(self):
        return 4

    def get_parameters_bounds(self):
        bounds = [
            (1, 5), #L2 norm params, 0
            (1, 200), #type1 neurons, 1 (1, 200)
            (1, 200), #type2 neurons, 2 (1, 200)
            (1, 2), #type5 neurons, 3
        ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        print ("######################################################################################")
        l2_parms_lst = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        l2_parm = l2_parms_lst[genotype[0]-1]
        type_neuron_lst = ["tanh", "sigm", "lin"]

        lin_check = genotype[3]

        num_neuron_lst = []

        for n in range(2):
            num_neuron_lst.append(genotype[n+1]*10)

        if lin_check == 1:
            num_neuron_lst.append(20)
        else:
            num_neuron_lst.append(0)

        print("l2_params: " ,l2_parm)
        print ("lin_check: ", lin_check)
        print("num_neuron_lst: ", num_neuron_lst)
        print("type_neuron_lst: ", type_neuron_lst)

        feat_len = self.train_sample_array.shape[1]

        elm_class = network_fit(feat_len, l2_parm, lin_check,
                                num_neuron_lst, type_neuron_lst, self.model_path, self.device, self.batch)

        elm_net = elm_class.trained_model()
        validation = elm_class.train_net(elm_net, self.train_sample_array, self.train_label_array, self.val_sample_array,
                                    self.val_label_array)

        val_value = validation[0]

        print ("num_neuron_lst", num_neuron_lst)
        penalty = self.constant * sum(num_neuron_lst)

        val_penalty = val_value + penalty

        val_penalty = round(val_penalty, 8)
        val_value = round(val_value, 8)

        print ("validation rmse-%s, penalty-%s, num_neurons-%s, const-%s" %(str(val_value), str(penalty),
                                                                            str(self.constant), str(sum(num_neuron_lst))))


        if self.obj == "soo":
            # fitness = (val_penalty,)
            fitness = (val_value,)
        elif self.obj == "moo":
            fitness = (val_value, sum(num_neuron_lst))

        print("fitness: ", fitness)

        elm_class = None
        elm_net  = None
        del elm_class, elm_net

        return fitness


