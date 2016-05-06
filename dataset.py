import numpy as np
import time
t0 = time.time() 
import theano 
#print("Took {:.2f} seconds to load theano".format(time.time() - t0))
import h5py 
import os
import cPickle
import shutil

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
        if "mnist" in filename:
            print("Using MNIST dataset") 
            arrays = self.loadmnist(filename)
        else:
            arrays = self.loadDataSet(filename) 
        self.train_x = arrays[0]
        self.train_y = arrays[1]
        self.test_x = arrays[2]
        self.test_y = arrays[3]
        self.indexing = arrays[4]
        self.trainset_size = self.train_x.get_value().shape[0]
        self.testset_size = self.test_y.get_value().shape[0]
        self.filename = filename
        if name == None:
            self.name = os.path.splitext(filename)[0]
        else:
            self.name = name 
        self.vec_size = self.train_x.get_value().shape[1]

    def __getitem__(self,name):
        """
        Get one of the hdf5 data set attributes
        """
        if ".pkl" in self.filename:
            print("This method doesn't work for pickle files") 
            return None
        try:
            val = self.dataset.attrs[name]
            return val 
        except KeyError:
            print("Dataset doesn't have that attribute") 
            print("Here's a list of available attributes:")
            for key in self.dataset.attrs.keys():
                print(key)
            return None 
        except AttributeError:
            print("Dataset hasn't been initialized yet!") 

    def reshuffle(self):
        """
        Reshuffle training and testing sets. 
        """
        print("Reshuffling dataset")
        t0 = time.time()  
        indexing = self.indexingset[...] 
        np.random.shuffle(indexing)  
        self.indexingset[...] = indexing 
        #now close and reopen file 
        self.f.close() 
        self.f = h5py.File(self.filename,'r+')
        self.dataset = self.f['waveforms'] 
        self.indexingset = self.f['indexing']
        dat = self.dataset[...] 
        indexing = self.indexingset[...]
        train_ind = indexing[:int(0.6*dat.shape[0])]
        test_ind = indexing[int(0.6*dat.shape[0]):]
        train = dat[train_ind]
        test = dat[test_ind] 
        train_x = theano.shared(train[:,1:], borrow=True)
        train_y = theano.shared(train[:,0].astype(int),borrow=True)
        test_x = theano.shared(test[:,1:],borrow=True)
        test_y = theano.shared(test[:,0].astype(int),borrow=True) 
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        print("Completed reshuffling. Took {} seconds".format(time.time() - t0))
 
    def loadmnist(self,filename): 
        """
        MNIST is structured as follows:
        (train, test, valid) 
        """
        print("Loading MNIST dataset...") 
        t0 = time.time()  
        with open(filename, 'r') as f:
            arrays = cPickle.load(f)
        print("Took {:.2f} to load dataset".format(time.time() - t0))
        train = arrays[0]
        test = arrays[1] 
        train_x = theano.shared(np.asarray(train[0],dtype=float), borrow=True)
        train_y = theano.shared(np.asarray(train[1],dtype=int), borrow=True)
        test_x = theano.shared(np.asarray(test[0],dtype=float), borrow=True)
        test_y = theano.shared(np.asarray(test[1],dtype=int), borrow=True)
        #print(train_x.get_value().shape) 
        #print(test_y.get_value().shape)
        #print(test_y.get_value())
        return (train_x, train_y, test_x, test_y)  

    def loadDataSet(self,filename):
        t0 = time.time() 
        print("Loading data set...") 
        f = h5py.File(filename,'r+') 
        self.f = f 
        self.dataset = f['waveforms']
        try:
            self.indexingset = f["indexing"]
        except KeyError:
            f.create_dataset("indexing",data=np.arange(self.dataset[...].shape[0],dtype=int))
            self.indexingset= f["indexing"] 
        indexing = self.indexingset[...]
        dat = self.dataset[:,:]
        train_ind = indexing[:int(0.6*dat.shape[0])]
        test_ind = indexing[int(0.6*dat.shape[0]):] 
        #np.random.shuffle(dat) 
        train = dat[train_ind]
        test = dat[test_ind] 
        train_x = theano.shared(train[:,1:], borrow=True)
        train_y = theano.shared(train[:,0].astype(int),borrow=True)
        test_x = theano.shared(test[:,1:],borrow=True)
        test_y = theano.shared(test[:,0].astype(int),borrow=True)
        print("Dataset loaded. Took {:.2f} seconds".format(time.time() - t0))
        return (train_x, train_y, test_x, test_y,indexing)


if __name__ == '__main__':
    #data_file = "dataFiles/datPS_10000_02-05_norm_by-wf_ignoreTop.hdf5"
    #dataset = Dataset(data_file) 
    mnist_file = "dataFiles/mnist.pkl"
    dataset = Dataset(mnist_file) 
