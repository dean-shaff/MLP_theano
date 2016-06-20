import theano
import theano.tensor as T 
import numpy as np 
import time 
import os 
from dataset import Dataset
from MLP import MLP 
import matplotlib.pyplot as plt 
import seaborn as sns 
#import sklearn.metrics as metrics 
import re 
import h5py
import argparse 

real_names = {
    'by-wf':'Waveform approach',
    'by-chan':'Channel approach',
    'all':'All PMT criterion',
    'top':'Top PMT criterion',
    'bottom': 'Bottom PMT criterion',
    '12back10sig': 'run 12 as background, run 10 as signal'
}


def gen_parser():
    parser = argparse.ArgumentParser(description="Sample from a trained MLP")
    parser.add_argument("-mf","--modelfiles",action="store",dest="mf",nargs="+",
                        help="Specify full path to model files",required = True)
    results = parser.parse_args()
    print(results.mf) 
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
        self.dataset.set_indexing(self.param['indexing'])
        self.shared_train_x = self.dataset.train_x
        self.shared_train_y = self.dataset.train_y 
        self.shared_test_x = self.dataset.test_x
        self.shared_test_y = self.dataset.test_y
        try:
            self.train_labels = self.dataset.train_labels
            self.test_labels = self.dataset.test_labels
        except AttributeError:
            print("You're used a dataset without labels. You won't be able to call gen_labeled_outputs")  
        mlp = MLP(x,[self.param['h0'],self.param['h1'],2],np.random.RandomState(1234), transfer_func=T.nnet.relu)
        mlp.load_params(self.model_file,mode='hdf5')
        self.model = mlp   
        self.predicted = dict()

    def compileFunctions(self,x,y):
        """
        Compile Theano functions.
        test -- get the error of a certain segment of the dataset 
        feed_thru -- feed a dataset (or a segment thereof) through the MLP, to generate predictions. 
        """

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
   
    def gen_labeled_outputs(self, save=True):
        """
        Generate a correspondance between dataset labels and MLP output. 
        kwargs:
            save: Save labels, signal or background category and MLP output value to a file 
        """
        t0 = time.time()
        cur_time = time.strftime("%d-%m")  
        try:
            feed_thru = self.feed_thru
        except AttributeError:
            print("You didn't compile relevanet theano functions yet. Doing it now...") 
            x = T.matrix('x') 
            y = T.lvector('y') 
            self.compileFunctions(x,y)
            feed_thru = self.feed_thru
        try:
            train_labels = self.train_labels.get_value() 
            test_labels = self.test_labels.get_value() 
        except AttributeError:
            print("The dataset you're using doesn't have labels.") 
            return  
        
        train_output = feed_thru(self.shared_train_x.get_value())[:,0]
        test_output = feed_thru(self.shared_test_x.get_value())[:,0]
        f = h5py.File("labeled_output_{}.hdf5".format(cur_time),"w")
        grpmain = f["/"] 
        grpmain.attrs['modelfile'] = self.model_file 
        grptrain = f.create_group('train') 
        grptest = f.create_group('test') 
        grptrain.create_dataset('labels',data=train_labels) 
        grptrain.create_dataset('output',data=train_output) 
        grptrain.create_dataset('category',data=self.shared_train_y.get_value()) 
         
        grptest.create_dataset('labels',data=test_labels) 
        grptest.create_dataset('output',data=test_output) 
        grptest.create_dataset('category',data=self.shared_test_y.get_value())
        f.close()  
        print("Time generating labeled output: {:.2f} seconds".format(time.time() - t0))
    
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
 
    def plot_distributions(self,which,ax,ax1):
        try:
            output_list = self.predicted[which]
        except KeyError:
            self.gen_output_distribution()
            output_list = self.predicted[which]
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
        ax1.plot(tpr, fpr,lw=2,label="{}, {}, Integral Value: {:.2f}".format(real_names[self.dataset['approach']], real_names[self.dataset['criterion']],integral))
        ax.legend(fontsize=20)
        ax.set_title("{}, {},  error {:.3f}".format(real_names[self.dataset['approach']], real_names[self.dataset['criterion']],float(err)),fontsize=14) 
        ax1.set_title("ROC Curves" ,fontsize=20)
        ax1.legend(fontsize=20,loc=4) 
        ax1.set_xlabel("True Positive Rate",fontsize=20) 
        ax1.set_ylabel("False Positive Rate",fontsize=20) 
            #ax.set_yscale('log')
        #plt.show() 

    def detect_params(self,model_file):
        f = h5py.File(model_file)
        params = {}
        grp = f["/"]
        for key in grp.attrs.keys():
            params[key] = grp.attrs[key]
        try:
            params['indexing'] = f['indexing'][...] 
        except KeyError:
            print("Using older model file -- doesn't have indexing yet.") 
            params['indexing'] = None 
        f.close() 
        print(params) 
        return params

    @staticmethod(wf_file, score_file):
        """
        Generate a correspondance between waveforms and MLP output score  
        """    
    

 
if __name__ == "__main__":
    x = T.matrix('x')
    y = T.lvector('y')
    #model_files = "modelFiles/modelBEST_epoch_102_mb_50_lr_0.005_h0_100_hin_1140_04-05.hdf5"
    #model_files = "modelFiles/modelBEST_epoch_466_mb_50_lr_0.005_h0_300_hin_1140_04-05.hdf5"
    #model_files = "modelFiles/modelBEST_epoch_194_mb_20_lr_0.001_h0_500_hin_1140_05-05.hdf5"
    #modelFile = "modelFiles/modelBEST_mb_50_lr_0.003_h0_300_hin_1140_05-05.hdf5"
    result = gen_parser()
    fig = plt.figure(figsize=(9,4*len(result.mf)))
    fig1 = plt.figure(figsize=(16,9))
    ax1 = fig1.add_subplot(111) 
    for i,mf in enumerate(result.mf):
        ax = fig.add_subplot(int(str(len(result.mf))+"1"+str(i+1)))
        sampler  = Sampler(x,mf)
        sampler.compileFunctions(x,y) 
        sampler.gen_labeled_outputs(save=True) 
        #sampler.plot_distributions('test',ax,ax1)
    #fig.suptitle("Output distributions for Signal and Background",fontsize=20) 
    #fig.set_tight_layout({'rect':[0,0.03,1,0.95]}) 
    #fig.savefig("outputdistr.png")
    #fig1.savefig("roc.png") 
