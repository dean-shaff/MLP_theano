import numpy as np 
import theano
import theano.tensor as T 
from MLP import MLP, HiddenLayer 
from dataset import Dataset

class SGD(object):

    def __init__(self, model, dataset):

        self.model = model 
        self.dataset = dataset 
    
    def compileFunctions(self):
        pass

    def trainModel(**kwargs):

        lr = kwargs.get('lr',0.001) 
        bs = kwargs.get('bs',100) 


