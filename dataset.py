import numpy as np
import time
t0 = time.time() 
import theano 
#print("Took {:.2f} seconds to load theano".format(time.time() - t0))
import h5py 
import os

class Dataset(object):

    def __init__(self,filename,name=None):
        """
        initialize a data set, with training and testing sets.
        args:
            arrays:  a list of theano shared variables
            filename: name of file used to generate dataset 
        kwargs:
            name: name of dataset (None)
        """
        arrays = self.loadDataSet(filename) 
        self.train_x = arrays[0]
        self.train_y = arrays[1]
        self.test_x = arrays[2]
        self.test_y = arrays[3]
        self.trainset_size = self.train_x.get_value().shape[0]
        self.testset_size = self.test_y.get_value().shape[0]
        self.filename = filename
        if name == None:
            self.name = os.path.splitext(filename)[0]
        else:
            self.name = name 
        self.vec_size = self.train_x.get_value().shape[1]

    def reshuffle(self):
        pass

    def loadDataSet(self,filename):
        t0 = time.time() 
        print("Loading data set...") 
        f = h5py.File(filename,'r') 
        train = f["train"][:,:]
        test = f["test"][:,:]
        train_x = theano.shared(train[:,1:], borrow=True)
        train_y = theano.shared(train[:,0].astype(int),borrow=True)
        test_x = theano.shared(test[:,1:],borrow=True)
        test_y = theano.shared(test[:,0].astype(int),borrow=True)
        print("Dataset loaded. Took {:.2f} seconds".format(time.time() - t0))
        return (train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    data_file = "dataFiles/datPS_10000_02-05_norm_by-wf_ignoreTop.hdf5"
    dataset = Dataset(data_file) 
