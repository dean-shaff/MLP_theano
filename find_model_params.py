import h5py
import argparse

def find_params(modelfile):
    f = h5py.File(modelfile,'r') 
    grp = f["/"]
    params = {}
    for key in grp.attrs.keys():
        params[key] = grp.attrs[key]
    indexing = f['indexing'][...]
    return params, indexing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Figure out a model's dataset") 
    parser.add_argument("-mf","--modelfile",action='store',dest='mf',
                        help="Specify full path to model file", required=True) 
    results = parser.parse_args()
    mf = results.mf
    print(find_params(mf))

