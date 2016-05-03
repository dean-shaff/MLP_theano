import theano
import theano.tensor as T 
import numpy as np 
import time 
import os 
from dataset import Dataset
from MLP import MLP 
import matplotlib.pyplot as plt 

class Sampler(object):
    
    def __init__(self, model, dataset, model_files):
        """
        Sampling works as follows.
        You feed it a model and a dataset, and model_files. 
        The model_files allow you to load in models from different checkpoints.
        A feed through function is compiled that samples from the test dataset. 
        It calculates the error and the output for each element in test dataset. 
        It generates two distributions -- output for signal, and output for background.
        args:
            model: MLP object
            dataset Dataset object
            model_files: list of files corresponding to saved models  
        """ 
        self.model = model
        self.dataset = dataset 
        self.shared_train_x = dataset.train_x
        self.shared_train_y = dataset.train_y 
        self.shared_test_x = dataset.test_x
        self.shared_test_y = dataset.test_y 
        self.model_files = model_files 
    def compileFunctions(self,x,y):

        index = T.lscalar() 
        self.errors = self.model.errors(y) 
        print("Compiling functions...") 

        t0 = time.time() 
        self.feed_thru = theano.function(
            inputs=[x],
            outputs=self.model.MLPoutput,
            #givens = {
            #    x: self.shared_test_x
            #}
        )
        self.test = theano.function(
            inputs=[x,y],
            outputs = self.errors,
            #givens = {
            #    x:self.shared_test_x,
            #    y:self.shared_test_y
            #}
        )
        print("Function compilation complete. Took {:.2f} seconds".format(time.time() -t0)) 
    
    def gen_output_distribution(self,**kwargs):
        """
        Generate output distributions for model files.
        kwargs:
            -plot: plot the output distributions (False) 
        """ 
        try:
            feed_thru = self.feed_thru
            test = self.test 
        except AttributeError:
            print("You didn't call compile functions yet. Doing it now...") 
            x = T.matrix('x') 
            y = T.lvector('y') 
            self.compileFunctions(x,y) 
            feed_thru = self.feed_thru
            test = self.test 
        plot = kwargs.get('plot',False) 
        signal = [] 
        background = [] 
        signal_train = [] 
        background_train = [] 
        errors_test = []
        errors_train = []  
        test_y, train_y = self.shared_test_y.get_value(), self.shared_train_y.get_value()
        sig_loc = np.where(test_y == 1) 
        back_loc = np.where(test_y == 0)
        sig_loc_train = np.where(train_y == 1) 
        back_loc_train = np.where(train_y == 0) 
        for filename in self.model_files:
            self.model.load_params(filename,mode='hdf5') 
            sampled = feed_thru(self.shared_test_x.get_value())
            sampled_train = feed_thru(self.shared_train_x.get_value()) 
            error_test = test(self.shared_test_x.get_value(), self.shared_test_y.get_value())
            error_train = test(self.shared_train_x.get_value(), self.shared_train_y.get_value())
            errors_test.append(error_test) 
            errors_train.append(error_train) 
            signal.append(sampled[sig_loc]) 
            background.append(sampled[back_loc])
            signal_train.append(sampled_train[sig_loc_train]) 
            background_train.append(sampled_train[back_loc_train]) 
        if plot:
            fig = plt.figure(figsize=(16,9)) 
            ax = fig.add_subplot(111) 
            for sig, back in zip(signal, background):
                sig = sig[:,1]
                back = back[:,1]
                w_sig = np.ones_like(sig) / len(sig) 
                w_back = np.ones_like(back) / len(back) 
                ax.hist(sig,50, weights=w_sig,facecolor='r',alpha=1.0 )
                ax.hist(back,50, weights=w_back,facecolor='k',alpha=0.5)
                #ax.set_yscale('log') 
            plt.show() 

        return signal, background 
         
 
if __name__ == "__main__":
    dataFile = "dataFiles/datPS_10000_02-05_norm_by-wf_ignoreTop.hdf5"
    dataset = Dataset(dataFile)
    x = T.matrix('x')
    y = T.lvector('y')
    model = MLP(x, [1140,200,2],np.random.RandomState(1234),transfer_func=T.nnet.relu)
    model_files = ["modelFiles/model_epoch980.hdf5"]
    sampler  = Sampler(model,dataset,model_files)
    sampler.compileFunctions(x,y) 
    sampler.gen_output_distribution(plot=True)  
