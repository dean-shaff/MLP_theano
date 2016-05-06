import h5py
import argparse

def find_dataset(modelfile):
    f = h5py.File(modelfile,'r') 
    grp = f["/"]
    return grp.attrs['dataset']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Figure out a model's dataset") 
    parser.add_argument("-mf","--modelfile",action='store',dest='mf',
                        help="Specify full path to model file", required=True) 
    results = parser.parse_args()
    mf = results.mf
    print(find_dataset(mf))

