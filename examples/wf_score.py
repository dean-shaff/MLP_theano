import sys 
sys.path.append("/Users/dean/capstone/MLPtheano/") 
from MLP_Theano import Sampler 
import matplotlib.pyplot as plt 

score_file = "/Users/dean/capstone/MLPtheano/labeled_output_20-06.hdf5" 
wf_file = "/Users/dean/capstone/MLPtheano/dataFiles/wfsPS_24000_20-06_norm_by-wf_all.hdf5" 

if __name__ == '__main__':
    ds_type = 'top' 
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    gen = Sampler.view_waveforms(wf_file, score_file, max_wf=10000, generator=True)
    num_sig = 0
    num_back = 0 
    for lab, score, wf in gen:
        if num_sig >= 100 and num_back >= 100:
            break 
        if score < 0.2:
            num_sig += 1
            print("Signal count: {}".format(num_sig))  
            ax.plot(wf, drawstyle='steps-mid') 
            ax.set_xlim([0,wf.shape[0]]) 
            ax.set_title("MLP score: {:.2f}".format(score),fontsize=20) 
            ax.set_xlabel("Time (10 ns)") 
            ax.set_ylabel("ADC counts above baseline")
            fig.savefig("./../plots/wf_sig_{}_{:04}.png".format(ds_type, num_sig))
            plt.pause(0.01)
            ax.clear()  
        if score > 0.7:       
            num_back += 1 
            print("Background count: {}".format(num_back))  
            ax.plot(wf, drawstyle='steps-mid') 
            ax.set_xlim([0,wf.shape[0]]) 
            ax.set_title("MLP score: {:.2f}".format(score),fontsize=20) 
            ax.set_xlabel("Time (10 ns)") 
            ax.set_ylabel("ADC counts above baseline")
            fig.savefig("./../plots/wf_back_{}_{:04}.png".format(ds_type,num_back))
            plt.pause(0.01)
            ax.clear()  
