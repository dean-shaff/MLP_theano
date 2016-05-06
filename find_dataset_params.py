import h5py
import argparse 

def find_params(df):
    f = h5py.File(df, 'r') 
    dataset = f['waveforms'] 
    params = {} 
    for key in dataset.attrs.keys():
        params[key] = dataset.attrs[key]
    f.close()  
    return params 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure out dataset parameters') 
    parser.add_argument("-df","--datafile",action='store', dest='df',
                        help = "Specify full path to data file", required=True) 
    results = parser.parse_args()
    print(find_params(results.df))
