import numpy as np 
import theano
import theano.tensor as T 
from MLP import MLP, HiddenLayer 
from dataset import Dataset
import time

class SGD(object):

    def __init__(self, model,dataset):

        self.model = model 
        self.dataset = dataset 
        self.shared_train_x = dataset.train_x
        self.shared_train_y = dataset.train_y
        self.shared_test_x = dataset.test_x
        self.shared_test_y = dataset.test_y

    def compileFunctions(self,x,y,**kwargs):
        """
        Compile testing and training functions. Defines hyperparameters symbolically.
        Hyperparams:
            -learning rate: lr 
            -minibatch size: mb_size 
            -momentum: momentum
        """ 
        # define hyperparameters symbolically. 
        lr = T.scalar("lr",dtype=theano.config.floatX) 
        mb_size = T.scalar('mb_size',dtype='int64')
        index = T.scalar("index",dtype='int64')
        momentum = T.scalar("momentum",dtype=theano.config.floatX)  
        self.cost = self.model.negloglikelihood(y)
        self.errors = self.model.errors(y)  
        print("Starting to compile theano functions. Could take a bit...") 
        t0 = time.time()
        updates = [] 
        for param in self.model.params:
            param_update = theano.shared(param.get_value()*0.) 
            updates.append((param, param-lr*param_update))
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(self.cost,param)))

        #gparams = [T.grad(self.cost,param) for param in self.model.params]
        #updates = [(param, param-lr*gparam) for param, gparam in zip(self.model.params, gparams)] 
        self.train_model = theano.function(
            inputs = [index,lr,mb_size,momentum],
            outputs = self.cost,
            updates = updates,
            givens = {
                x: self.shared_train_x[index*mb_size: (index+1)*mb_size],
                y: self.shared_train_y[index*mb_size: (index+1)*mb_size]
            }
        )
        #self.feed_thru = theano.function(
        #    inputs = [index],
        #    outputs = self.model.hiddenLayers[-1].output,
        #    givens = {
        #        x: self.shared_train_x[index*mb_size: (index+1)*mb_size]#,
        #       # y: self.shared_y[index*mb_size: (index+1)*mb_size]
        #    }
        #)
        #print("Done compiling feed through function") 
        self.test = theano.function(
            inputs = [x,y],
            outputs = self.errors
        )
        #self.test_on_train = theano.function(
        #    inputs = [index],
        #    outputs = self.errors,
        #    givens = {
        #        x: self.shared_train_x[index*mb_size: (index+1)*mb_size],
        #        y: self.shared_train_y[index*mb_size: (index+1)*mb_size]
        #    }
        #)
        print("Functions compiled. Took {:.2f} seconds".format(time.time() - t0))
    def trainModel(self,**kwargs):
        """
        train the model using stochastic Gradient Descent. 
        You have to explicitly pass it parameters.  
        kwargs:
            n_epochs: number of epochs
            save_rate: How often to save model parameters (20) 
                if None, doesn't save 
            test_rate: How often to test the model 
            lr: learning rate (None) 
            mb_size: minibatch size (None) 
            momentum: momentum (None) 
        """
        n_epochs = kwargs.get('n_epochs',100)
        save_rate = kwargs.get('save_rate',20) 
        test_rate = kwargs.get('test_rate',5)
        self.mb_size = kwargs.get('mb_size',None) 
        self.lr = kwargs.get('lr',None) 
        self.momentum = kwargs.get('momentum',None)  
        try:
            trainer = self.train_model
            tester = self.test
        except AttributeError:
            print("You haven't called compile functions yet!")
            print("Calling compile functions with learning rate 0.001 and mini batch size 100")
            self.compileFunctions()
            trainer = self.train_model
            tester = self.test
            tester_on_train = self.test_on_train 
        
        if save_rate == None:
            save = False
        else:
            save = True  
        train_batches = self.dataset.trainset_size // self.mb_size   
        test_batches = self.dataset.testset_size // self.mb_size   
        print("Beginning to train MLP") 
        t0 = time.time()
        lowest_error = 0.6 # if its ever this high kill me 
        for epoch in xrange(n_epochs):
            t1 = time.time() 
            for mb_index in xrange(train_batches):
                #trainer(index,lr,mb_size,momentum) 
                cur_cost = trainer(mb_index,self.lr,self.mb_size,self.momentum)             
            print("Time for epoch {}: {:.2f}".format(epoch,time.time() - t1))
            if (epoch % test_rate == 0):
                #error_test = 0
                #error_train = 0 
                #for test_index in xrange(test_batches):
                #    error_test +=  tester(test_index)
                #for train_index in xrange(train_batches):
                #    error_train += tester_on_train(train_index)
                error_test = tester(self.shared_test_x.get_value(),self.shared_test_y.get_value()) 
                error_train = tester(self.shared_train_x.get_value(), self.shared_train_y.get_value()) 
                #error_test /= test_batches
                #error_train /= train_batches 
                print("Current test error: {}".format(error_test))
                print("Current train error: {}".format(error_train))
                if error_test < lowest_error:
                    lowest_error = error_test         
                    if (save):
                        cur_time = time.strftime("%d-%m")
                        self.model.save_params("modelFiles/modelBEST_mb_{}_lr_{}_mom_{}_h0_{}_hin_{}_{}.hdf5".format(self.mb_size,self.lr,self.momentum,self.model.dim[1],self.model.dim[0],cur_time),self.dataset.indexing,mode='hdf5',lr=self.lr, mb=self.mb_size,momentum=self.momentum, epoch=epoch,dataset=self.dataset.filename)
                        print("\n")
        self.dataset.f.close() 
         
    
if __name__ == "__main__":
    #dataFile = "dataFiles/datPS_20000_04-05_norm_by-wf_bottom.hdf5"
    #dataFile = "dataFiles/datPS_10000_05-05_norm_by-chan_bottom.hdf5"
    dataFile = "dataFiles/datPS_36000_06-05_norm_by-wf_top.hdf5"
    #dataFile = "dataFiles/datPS_24000_05-05_norm_by-wf_top.hdf5"
    dataset = Dataset(dataFile)
    dataset.reshuffle() 
    x = T.matrix('x')
    y = T.lvector('y')
    model = MLP(x,[dataset.vec_size,300,2],np.random.RandomState(1234),transfer_func=T.nnet.relu)
    sgd = SGD(model,dataset)
    sgd.compileFunctions(x,y)
    sgd.trainModel(n_epochs=200,test_rate=2,lr=0.005,momentum=0.9,mb_size=20) 

    

    #mnist_file = "dataFiles/mnist.pkl"
    #dataset_mnist = Dataset(mnist_file) 
    #x = T.matrix('x')
    #y = T.lvector('y') 
    #model = MLP(x, [dataset_mnist.vec_size,500,10],np.random.RandomState(1234))
    #sgd = SGD(model,dataset_mnist)
    #sgd.compileFunctions(x,y,lr=0.01,mb_size=20) 
    #sgd.trainModel(n_epochs=100)
         
            
     


