import theano
import theano.tensor as T 
import numpy as np 
import time 
import os 
from dataset import Dataset
from MLP import MLP 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn.metrics as metrics 
import re 
import h5py
import argparse 

def gen_parser():
    parser = argparse.ArgumentParser(description="Sample from a trained MLP")
    parser.add_argument("-mf","--modelfile",action="store",dest="mf",
                        help="Specify full path to model file",required = True)
    return parser.parse_args() 

class Sampler(object):
    
    def __init__(self,x, model_file):
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
        self.model_file = model_file
        self.param = self.detect_params(self.model_file)
        self.dataset = Dataset(self.param['dataset'])
        self.shared_train_x = self.dataset.train_x
        self.shared_train_y = self.dataset.train_y 
        self.shared_test_x = self.dataset.test_x
        self.shared_test_y = self.dataset.test_y 
        mlp = MLP(x,[self.param['h0'],self.param['h1'],2],np.random.RandomState(1234), transfer_func=T.nnet.relu)
        mlp.load_params(self.model_file,mode='hdf5')
        self.model = mlp   
        self.predicted = dict()
    def compileFunctions(self,x,y):

        index = T.lscalar() 
        self.errors = self.model.errors(y) 
        print("Compiling functions...") 

        t0 = time.time() 
        self.feed_thru = theano.function(
            inputs=[x],
            outputs=self.model.MLPoutput,
        )
        self.test = theano.function(
            inputs=[x,y],
            outputs = self.errors,
        )
        print("Function compilation complete. Took {:.2f} seconds".format(time.time() -t0)) 
    
    def gen_output_distribution(self,**kwargs):
        """
        Generate output distributions for model files.
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
 
        def gen_sig_back(x,y):
            """
            x and y are the numpy arrays (not theano shared variables) corresponding to 
            data and labels.
            """
            sig_loc = y == 1
            back_loc = y == 0 
            sampled = feed_thru(x)
            error = test(x,y)
            signal = sampled[sig_loc]
            background = sampled[back_loc]
            total = sampled 
            return [signal,background,total,error,y]

        test_y, train_y = self.shared_test_y.get_value(), self.shared_train_y.get_value()
        test_x, train_x = self.shared_test_x.get_value(), self.shared_train_x.get_value() 
        self.predicted['train'] = gen_sig_back(train_x, train_y) 
        self.predicted['test'] = gen_sig_back(test_x, test_y)
 
    def plot_distributions(self,which):
        try:
            output_list = self.predicted[which]
        except KeyError:
            self.gen_output_distribution()
            output_list = self.predicted[which]
        fig = plt.figure(figsize=(16,9)) 
        fig1 = plt.figure(figsize=(16,9)) 
        ax = fig.add_subplot(111) 
        ax1 = fig1.add_subplot(111)
        sig, back, total,err,y = output_list
        
        sig = sig[:,0]
        back = back[:,0]
        tot = total[:,0]
        w_sig = np.ones_like(sig) / len(sig) 
        w_back = np.ones_like(back) / len(back) 
        ax.hist(sig,50, weights=w_sig,facecolor='r',alpha=1.0,label='Signal')
        ax.hist(back,50, weights=w_back,facecolor='k',alpha=0.35,label='Background')
        fpr, tpr,_ = metrics.roc_curve(y, tot)
        integral = np.dot(np.diff(tpr), fpr[:-1])  
        ax1.plot(tpr, fpr,lw=2,label="Integral Value: {:.2f}".format(integral))
        ax.legend(fontsize=20)
        ax.set_title("Output distributions for signal and background\nlearning rate: {}, hidden layer size: {}, minibatch size: {}, error {:.3f}".format(self.param['lr'],self.param['h0'],self.param['mb'],float(err)),fontsize=20) 
        ax1.set_title("ROC Curve" ,fontsize=20)
        ax1.legend(fontsize=20,loc=4) 
        ax1.set_xlabel("True Positive Rate",fontsize=20) 
        ax1.set_ylabel("False Positive Rate",fontsize=20) 
        plt.tight_layout()
            #ax.set_yscale('log')
        plt.show() 

    def detect_params(self,model_file):
        #name = os.path.splitext(model_file)[0]
        #names = name.split("_")
        #lr = float([names[i+1] for i in xrange(len(names)) if names[i]=='lr'][0]) 
        #mb = int([names[i+1] for i in xrange(len(names)) if names[i]=='mb'][0])
        #h0 = int([names[i+1] for i in xrange(len(names)) if names[i]=='h0'][0])
        #hin = int([names[i+1] for i in xrange(len(names)) if names[i]=='hin'][0])
        #return {'lr':lr,
        #        'mb': mb,
        #        'h0': h0,
        #        'hin': hin}
        f = h5py.File(model_file)
        params = {}
        grp = f["/"]
        for key in grp.attrs.keys():
            params[key] = grp.attrs[key]
        f.close() 
        print(params) 
        return params
        

 
if __name__ == "__main__":
    #dataFile = "dataFiles/datPS_10000_02-05_norm_by-wf_ignoreTop.hdf5"
    #dataFile = "dataFiles/datPS_20000_04-05_norm_by-wf_ignoreTop.hdf5"
    dataFile = "dataFiles/datPS_20000_04-05_norm_by-wf_bottom.hdf5"
    dataFile = "dataFiles/datPS_10000_05-05_norm_by-chan_bottom.hdf5"
    #dataset = Dataset(dataFile)
    x = T.matrix('x')
    y = T.lvector('y')
    model_files = "modelFiles/modelBEST_epoch_102_mb_50_lr_0.005_h0_100_hin_1140_04-05.hdf5"
    model_files = "modelFiles/modelBEST_epoch_466_mb_50_lr_0.005_h0_300_hin_1140_04-05.hdf5"
    model_files = "modelFiles/modelBEST_epoch_194_mb_20_lr_0.001_h0_500_hin_1140_05-05.hdf5"
    modelFile = "modelFiles/modelBEST_mb_50_lr_0.003_h0_300_hin_1140_05-05.hdf5"
    result = gen_parser() 
    #model = MLP(x,[1140,100,2],np.random.RandomState(1234),transfer_func=T.nnet.relu)
    #model_files = ["modelFiles/model_epoch980.hdf5",
    #model_files = ["modelFiles/model_epoch40_mb20_lr0.005_04-05.hdf5"]
    #model_files = ["modelFiles/model_epoch200_mb20_lr0.005_h0500_hin1140_04-05.hdf5"]
    sampler  = Sampler(x,result.mf)
    sampler.compileFunctions(x,y) 
    #sampler.gen_output_distribution(plot=True)  
    sampler.plot_distributions('test')
     
