
import numpy
import theano

from numpy import zeros

class ScratchAutoencoder():

    def __init__(self, visible_size, hidden_size, a_rate, N_classes=2):

        self.X = theano.tensor.matrix("input data")
        self.y = theano.tensor.matrix("label")
        
        initial_We = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     high= numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     size=(visible_size,hidden_size)))
        # theano shared variables for weights and biases
        We = theano.shared(value=initial_We, name='We')
        be = theano.shared(value=numpy.zeros(hidden_size), name='be')
        
        initial_Wd = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     high= numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     size=(hidden_size, visible_size)))
        # theano shared variables for weights and biases
        Wd = theano.shared(value=initial_Wd, name='Wd')
        bd = theano.shared(value=numpy.zeros(visible_size), name='bd')
        
        initial_Wc = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(1. / (2 + hidden_size)),
                     high= numpy.sqrt(1. / (2 + hidden_size)),
                     size=(hidden_size,N_classes)))
        # theano shared variables for weights and biases
        Wc = theano.shared(value=initial_Wc, name='Wc')
        bc = theano.shared(value=numpy.zeros(N_classes), name='bc')
                
                
        
        z = theano.tensor.nnet.sigmoid(theano.tensor.dot(self.X,We)+be)
        recs = theano.tensor.dot(z,Wd)+bd

        X = theano.tensor.matrix()
        self._project = theano.function([X], z, givens = {self.X: X})
        self._reconstruct = theano.function([X], recs, givens = {self.X: X})
        
        Er_rec = ((self.X-recs)**2).sum(axis=1)
        h = theano.tensor.nnet.softmax(theano.tensor.dot(z,Wc)+bc)
        Er_clas = theano.tensor.nnet.categorical_crossentropy(h, self.y)


        XX = theano.tensor.matrix()
        YY = theano.tensor.matrix()
        self.learning_rate = theano.tensor.dscalar()
        index = theano.tensor.lscalar()
        
        params = [We,be,Wd,bd,Wc,bc]
        self.params = params
        cost1 = theano.tensor.mean(Er_rec + a_rate * Er_clas)
        grads1 = theano.tensor.grad(cost1, params)
        updates = [(param, param - self.learning_rate * grad) for grad,param in zip(grads1, params)]
        
        monitorize = [theano.tensor.mean(Er_rec), a_rate * theano.tensor.mean(Er_clas), cost1]
        self.f_train = theano.function([XX,YY,index], monitorize, updates=updates, givens = {self.X: XX[index*batch_size:(index+1)*batch_size,:], self.learning_rate: learning_rate, self.y: YY[index*batch_size:(index+1)*batch_size,:]}, on_unused_input='warn')
        self.f_monit = theano.function([XX, YY], monitorize, givens = {self.X: XX, self.y: YY}, on_unused_input='warn')

        params2 = [We,be,Wc,bc]
        cost_bp = theano.tensor.mean(a_rate * Er_clas)
        grads2 = theano.tensor.grad(cost_bp, params2)
        updates2 = [(param, param - self.learning_rate * grad) for grad,param in zip(grads2, params2)]
        self.f_train_backpropagation = theano.function([XX,YY,index], monitorize, updates=updates2, givens = {self.X: XX[index*batch_size:(index+1)*batch_size,:], self.learning_rate: learning_rate, self.y: YY[index*batch_size:(index+1)*batch_size,:]}, on_unused_input='warn')

        params3 = [We,be,Wd,bd]
        cost_aut = theano.tensor.mean(Er_rec)
        grads3 = theano.tensor.grad(cost_aut, params3)
        updates3 = [(param, param - self.learning_rate * grad) for grad,param in zip(grads3, params3)]
        self.f_train_autoencoder = theano.function([XX,YY,index], monitorize, updates=updates3, givens = {self.X: XX[index*batch_size:(index+1)*batch_size,:], self.learning_rate: learning_rate, self.y: YY[index*batch_size:(index+1)*batch_size,:]}, on_unused_input='warn')


        #self.monit2 = theano.function([XX,YY], [h,self.y, theano.tensor.nnet.categorical_crossentropy(h,self.y)], givens = {self.X : XX , self.y : YY})

    def train_backpropagation(self, train_data, labels, batch_size, learning_rate): 
        a = []
        for i in range(len(train_data)/batch_size):
            a.append(self.f_train_backpropagation(train_data,labels,i))        
        print str(numpy.mean(numpy.array(a),axis=0))
        return numpy.mean(numpy.array(a),axis=0)

    def train_autoencoder(self, train_data, labels, batch_size, learning_rate): 
        a = []
        for i in range(len(train_data)/batch_size):
            a.append(self.f_train_autoencoder(train_data,labels,i))        
        print str(numpy.mean(numpy.array(a),axis=0))
        return numpy.mean(numpy.array(a),axis=0)

    def train(self, train_data, labels, batch_size, learning_rate): 
        a = []
        for i in range(len(train_data)/batch_size):
            a.append(self.f_train(train_data,labels,i))        
        print str(numpy.mean(numpy.array(a),axis=0))
        return numpy.mean(numpy.array(a),axis=0)

        
    def project(self, data):
        return self._project(data)
        
    def reconstruct(self,data):
        return self._reconstruct(data)
    
    def train(self, train_data, labels, batch_size, learning_rate): 
        a = []
        for i in range(len(train_data)/batch_size):
            a.append(self.f_train(train_data,labels,i))        
        print str(numpy.mean(numpy.array(a),axis=0))
        return numpy.mean(numpy.array(a),axis=0)



import cPickle
from numpy import array

try:
    print str(len(X_train)) + "   " + str(len(X_valid))
except:
    import old_src.system.holter_io as holter_utils
    from numpy import array
    from dataset_utils.getData import getData
    from dataset_utils.GetTrainAndTestSets import getSignalsTrainTest
    cases = holter_utils.get_usable_cases('cardiosManager')
    ids_train, ids_test = getSignalsTrainTest(cases)
    beats_train, beats_test, labels_train, labels_test = getData(ids_train, ids_test)

    #f = open("../train_test_set")
    #train_set = cPickle.load(f)
    #valid_set = cPickle.load(f)
    #f.close()

    X_train, Y_train = beats_train, labels_train
    X_valid, Y_valid = beats_test, labels_test

    X_train = array(X_train)
    X_valid = array(X_valid)

    X_train /= 10
    X_valid /= 10


Y_train_def = zeros((len(Y_train),2))
for i in range(len(Y_train)):
    if Y_train[i] == 'V':
        Y_train_def[i][1]=1  
    else:
        Y_train_def[i][0]=1  

learning_rate = 0.01
ac = 100
batch_size = 128


visible_size = X_train.shape[1]
hidden_size = 250
pa = ScratchAutoencoder(visible_size, hidden_size, ac, N_classes = 2)
costs = []
#for epoch in range(10):
#    costs.append( pa.train(numpy.matrix(X_train,dtype=numpy.float64), numpy.matrix(Y_train_def,dtype=numpy.float64), batch_size, learning_rate))
#for epoch in range(10):
#    costs.append( pa.train_autoencoder(numpy.matrix(X_train,dtype=numpy.float64), numpy.matrix(Y_train_def,dtype=numpy.float64), batch_size, learning_rate))
for epoch in range(30):
    costs.append( pa.train_backpropagation(numpy.matrix(X_train,dtype=numpy.float64), numpy.matrix(Y_train_def,dtype=numpy.float64), batch_size, learning_rate))


visible_size2 = hidden_size
hidden_size2 = 10
pa2 = ScratchAutoencoder(visible_size2, hidden_size2, 1., N_classes = 2)
costs2 = []
#for epoch in range(10):
#    costs2.append( pa2.train(pa.project(numpy.matrix(X_train,dtype=numpy.float64)), numpy.matrix(Y_train_def,dtype=numpy.float64), batch_size, learning_rate))
#for epoch in range(10):
#    costs.append( pa2.train_autoencoder(pa.project(numpy.matrix(X_train,dtype=numpy.float64)), numpy.matrix(Y_train_def,dtype=numpy.float64), batch_size, learning_rate))
for epoch in range(10):
    costs.append( pa2.train_backpropagation(pa.project(numpy.matrix(X_train,dtype=numpy.float64)), numpy.matrix(Y_train_def,dtype=numpy.float64), batch_size, learning_rate))




import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
clf = KNeighborsClassifier(n_neighbors = 10)
clf.fit(pa2.project(pa.project(numpy.matrix(X_train,dtype=numpy.float64))), Y_train)
pred = clf.predict(pa2.project(pa.project(numpy.matrix(X_valid, dtype=numpy.float64))))
print classification_report(pred, Y_valid)

