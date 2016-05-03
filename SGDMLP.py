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
        Compile testing and training functions.
        kwargs:
            lr: learning rate (0.001)
            mb_size: minibatch_size (100)
            L1_reg: L1 regilarization (0)
            L2_reg: L2 regularization factor (0.0001)
        """ 
        lr = kwargs.get('lr',0.001) 
        mb_size = kwargs.get('mb_size',100)
        L1_reg = kwargs.get('L1_reg',0)
        L2_reg = kwargs.get('L2_reg',0.0001)
        self.mb_size = mb_size 
        self.cost = self.model.negloglikelihood(y) #+ L1_reg*self.model.L1 + L2_reg*self.model.L2_sqr
        self.errors = self.model.errors(y)  
        print("Starting to compile theano functions. Could take a bit...") 
        t0 = time.time()
        index = T.lscalar() 
        gparams = [T.grad(self.cost,param) for param in self.model.params]
        updates = [(param, param-lr*gparam) for param, gparam in zip(self.model.params, gparams)] 
        self.train_model = theano.function(
            inputs = [index],
            outputs = self.cost,
            updates = updates,
            givens = {
                x: self.shared_train_x[index*mb_size: (index+1)*mb_size],
                y: self.shared_train_y[index*mb_size: (index+1)*mb_size]
            }
        )
        print("Done compiling training function") 
        self.feed_thru = theano.function(
            inputs = [index],
            outputs = self.model.hiddenLayers[-1].output,
            givens = {
                x: self.shared_train_x[index*mb_size: (index+1)*mb_size]#,
               # y: self.shared_y[index*mb_size: (index+1)*mb_size]
            }
        )
        print("Done compiling feed through function") 
        self.test = theano.function(
            inputs = [index],
            outputs = self.errors,
            givens = {
                x: self.shared_test_x[index*mb_size: (index+1)*mb_size],
                y: self.shared_test_y[index*mb_size: (index+1)*mb_size]
            }
        )
        self.test_on_train = theano.function(
            inputs = [index],
            outputs = self.errors,
            givens = {
                x: self.shared_train_x[index*mb_size: (index+1)*mb_size],
                y: self.shared_train_y[index*mb_size: (index+1)*mb_size]
            }
        )
        print("Functions compiled. Took {:.2f} seconds".format(time.time() - t0))
    def trainModel(self,**kwargs):
        """
        train the model using stochastic Gradient Descent 
        kwargs:
            n_epochs: number of epochs
            save_rate: How often to save model parameters (20) 
                if None, doesn't save 
            test_rate: How often to test the model 
        """
        n_epochs = kwargs.get('n_epochs',100)
        save_rate = kwargs.get('save_rate',20) 
        try:
            trainer = self.train_model
            tester = self.test
            tester_on_train = self.test_on_train 
        except AttributeError:
            print("You haven't called compile functions yet!")
            print("Calling compile functions with learning rate 0.001 and mini batch size 100")
            self.compile_functions(**{'lr':0.001, 'mb_size':100})
            trainer = self.train_model
            tester = self.test
            tester_on_train = self.test_on_train 
        if save_rate == None:
            save = False
        else:
            save = True  
        test_rate = kwargs.get('test_rate',5) 
        train_batches = self.dataset.trainset_size // self.mb_size   
        test_batches = self.dataset.testset_size // self.mb_size   
        print("Beginning to train MLP") 
        t0 = time.time() 
        for epoch in xrange(n_epochs):
            t1 = time.time() 
            for mb_index in xrange(train_batches):
                cur_cost = trainer(mb_index)             
            #print("Current cost per vector: {:.5f}, epoch {}".format(cur_cost/mb_size, epoch)) 
            print("Time for epoch {}: {:.2f}".format(epoch,time.time() - t1))
            if (epoch % test_rate == 0):
                error_test = 0
                error_train = 0 
                for test_index in xrange(test_batches):
                    error_test +=  tester(test_index)
                for train_index in xrange(train_batches):
                    error_train += tester_on_train(train_index) 
                print("Current test error: {}".format(error_test/test_batches))
                print("Current train error: {}".format(error_train/train_batches))
            
            if (epoch % save_rate == 0 and save):
                self.model.save_params("modelFiles/model_epoch{}.hdf5".format(epoch),mode='hdf5')

    
if __name__ == "__main__":
    #dataFile = "dataFiles/datPS_10000_02-05_norm_by-wf_ignoreTop.hdf5"
    #dataset = Dataset(dataFile)
    #x = T.matrix('x')
    #y = T.lvector('y') 
    #model = MLP(x, [1140,400,2],np.random.RandomState(1234))
    #sgd = SGD(model,dataset)
    #sgd.compileFunctions(x,y,lr=0.001,mb_size=1000) 
    #sgd.trainModel(n_epochs=100)

    mnist_file = "dataFiles/mnist.pkl"
    dataset_mnist = Dataset(mnist_file) 
    x = T.matrix('x')
    y = T.lvector('y') 
    model = MLP(x, [dataset_mnist.vec_size,500,10],np.random.RandomState(1234))
    sgd = SGD(model,dataset_mnist)
    sgd.compileFunctions(x,y,lr=0.01,mb_size=20) 
    sgd.trainModel(n_epochs=100)
         
            
     


