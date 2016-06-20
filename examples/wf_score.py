import sys 
sys.path.append("/Users/dean/capstone/MLPtheano/") 
from MLP_Theano import Sampler 

score_file = "/Users/dean/capstone/MLPtheano/labeled_output_20-06.hdf5" 
wf_file = "/Users/dean/capstone/MLPtheano/dataFiles/wfsPS_24000_20-06_norm_by-wf_all.hdf5" 

if __name__ == '__main__':
    Sampler.view_waveforms(wf_file, score_file, max_wf=100) 
    
