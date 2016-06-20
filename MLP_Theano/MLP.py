import numpy as np 
import theano
import theano.tensor as T 
import time 
import cPickle as pickle
import h5py 

class HiddenLayer(object):

    def __init__(self, input, nIn, nOut,rng,transfer_func): 
        """
        args:
            input: symbolic network input 
            nIn: size of input vector
            nOut: size of output vector 
            rng: numpy random seed 
        """
        self.input = input
        self.nIn = nIn
        self.nOut = nOut
        self.W = theano.shared(
            value = np.asarray(
                        rng.uniform(low = -np.sqrt(6./(nIn + nOut)),
                                    high = np.sqrt(6./(nIn + nOut)),
                                    size = (nIn, nOut)),
                                    dtype=theano.config.floatX),
                        name='W',
                        borrow=True)
        self.b = theano.shared(
            value = np.zeros(nOut,).astype(theano.config.floatX),
                        name='b',
                        borrow=True)
        
        self.output = T.dot(input,self.W) + self.b 
        self.Toutput = transfer_func(self.output) 
        self.params = [self.W, self.b] 

    def set_params(self,params):
        """
        set the parameters of hidden layer. One might do this after saving.
        """
        self.W.set_value(params[0])
        self.b.set_value(params[1])
    
class MLP(object):
    
    def __init__(self, input, dim, rng,transfer_func=T.nnet.sigmoid): 
        """
        Define an MLP, consisting of hidden layers.
        args:
            -input: symbolic input for graph
            -dim: The dimensions of each of the layers in the NN 
            -rng: numpy random seed, for hidden layers
        """
        self.dim = dim  
        self.input = input 
        h0 = HiddenLayer(self.input, dim[0], dim[1], rng,transfer_func)
        hiddenLayers = [h0]
        params = h0.params         
        for i in xrange(1,len(dim)-1):
            h = HiddenLayer(hiddenLayers[-1].Toutput, dim[i], dim[i+1], rng,transfer_func) 
            hiddenLayers.append(h) 
            params += h.params 
        self.params = params 
        self.hiddenLayers = hiddenLayers 
        self.MLPoutput = T.nnet.softmax(self.hiddenLayers[-1].output)
        self.pred = T.argmax(self.MLPoutput, axis=1)
        if len(self.hiddenLayers) >= 2: 
            print("Haven't implemented L1/L2 regularization yet") 
        else:
            self.L1 = T.sum(abs(self.hiddenLayers[0].W)) + T.sum(abs(self.hiddenLayers[1].W))
            self.L2_sqr = T.sum(self.hiddenLayers[0].W**2) +T.sum(self.hiddenLayers[1].W**2)  
      
    def save_params(self,filename,indexing,**kwargs):
        """
        Save model parameters. Can save meta data about SGD parameters. 
        args:
            filename: where to save the file 
            indexing: the current indexing in the dataset to use 
        kwargs:
            mode: 'pickle' or 'hdf5'
            lr: the learning rate of SGD 
            mb: minibatch size 
            momentum: momentum factor for SGD 
            epoch: The saved epoch number
            dataset: The name of the dataset used for training  
        """
        print("Saving parameters....")
        mode = kwargs.get("mode","hdf5")
        lr = kwargs.get("lr","")
        mb = kwargs.get("mb","") 
        epoch = kwargs.get("epoch","")  
        dataset = kwargs.get("dataset","") 
        momentum = kwargs.get("momentum","") 
        t0 = time.time() 
        params = [param.get_value() for param in self.params]  
        if mode == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(params,f)
        elif mode == 'hdf5':
            f = h5py.File(filename,"w") 
            grp = f["/"]
            keys = ['lr','mb','momentum','epoch','dataset','h0','h1']
            vals = [lr, mb,momentum,epoch, dataset,self.dim[0],self.dim[1]]
            for i in xrange(len(keys)):
                grp.attrs[keys[i]] = vals[i]
            for i in xrange(len(self.hiddenLayers)):
                f.create_dataset("w{}".format(i), data=params[2*i])
                f.create_dataset("b{}".format(i), data=params[2*i + 1])
            f.create_dataset('indexing',data=indexing) 
            f.close() 
        print("Saving complete. Took {:.2f} seconds.".format(time.time() - t0))

    def load_params(self, filename, mode='pickle'):
        """
        Load and set model parameters from some checkpoint file.
        args:
            filename: file with save model parameters
        kwargs:
            mode: 'pickle' or 'hdf5' (hdf5 is between 1 and 2 orders of magnitude faster) 
        """
        print("Loading in model parameters....")
        t0 = time.time()
        if mode == 'pickle':
            with open(filename, 'r') as f:
                params = pickle.load(f) 
        elif mode == 'hdf5':
            f = h5py.File(filename, 'r')
            params = [] 
            for i in xrange(len(self.hiddenLayers)):
                params.append(f["w{}".format(i)][:,:])
                params.append(f["b{}".format(i)][:])
        self.set_params(params) 
        print("Loading complete. Took {:.2f} seconds.".format(time.time() - t0))

    def square_cost(self,y):
        """
        return square error fro a minibatch
        """
        return T.mean((self.MLPoutput - y)**2)

    def negloglikelihood(self,y):
        """
        return negative log likelihood
        """
        # to understand this one, think about numpy indexing... 
        return -T.mean(T.log(self.MLPoutput)[T.arange(y.shape[0]),y]) 
        
    def errors(self,y):
        """
        calculate error in a minibatch
        """ 
        return T.mean(T.neq(self.pred,y))

    def set_params(self,params):
        """
        Given a list of numpy arrays corresponding to parameters (weights and biases) 
        for layers, this will update model paramters.
        args:
            -params: a python list with numpy arrays corresponding to weights and biases
        """
        for i in xrange(len(self.hiddenLayers)):
            self.hiddenLayers[i].set_params(params[2*i:(2*i)+2])


if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    x = T.matrix('x') 
    mlp = MLP(x,[10,1000,1000,100],rng) 
    print([param.get_value().shape for param in mlp.params]) 
    mlp.save_params("test.hdf5",mode='hdf5')
    #mlp.save_params("test.pkl",mode='pickle')
    #mlp.load_params("test.pkl",mode="pickle")
    mlp.load_params("test.hdf5",mode="hdf5")
     
