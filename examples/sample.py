import sys 
sys.path.append("/Users/dean/capstone/MLPtheano/") 
from MLP_Theano import Sampler 
import theano.tensor as T 
import matplotlib.pyplot as plt 
import os 

os.chdir("/Users/dean/capstone/MLPtheano/") 

score_file = "/Users/dean/capstone/MLPtheano/labeled_output_20-06.hdf5" 
wf_file = "/Users/dean/capstone/MLPtheano/dataFiles/wfsPS_24000_20-06_norm_by-wf_all.hdf5" 
mf = "/Users/dean/capstone/MLPtheano/modelFiles/modelBEST_mb_50_lr_0.005_mom_0.0_h0_300_hin_150_20-06_all.hdf5" 


if __name__ == '__main__':
    x = T.matrix('x') 
    y = T.lvector('y') 
    fig = plt.figure(figsize=(9,4)) 
    fig1 = plt.figure(figsize=(16,9))
    ax1 = fig1.add_subplot(111)
    ax = fig.add_subplot(int("1"+"1"+str(1)))
    sampler  = Sampler(x,mf)
    sampler.compileFunctions(x,y)
    sampler.gen_labeled_outputs(save=True)
    sampler.plot_distributions('test',ax,ax1)
    fig.suptitle("Output distributions for Signal and Background",fontsize=20)
    fig.set_tight_layout({'rect':[0,0.03,1,0.95]})
    fig.savefig("outputdistr.png")
    fig1.savefig("roc.png") 
