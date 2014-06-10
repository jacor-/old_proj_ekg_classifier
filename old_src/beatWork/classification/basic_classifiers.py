
class __basic__():
    def _clf_executor(self,predict, test_beats):
        Z = []
        j = 0
        from numpy import array
        
        from mdp.utils import progressinfo
    
        if len(test_beats) == 0:
            return array([])

        last = 0
        for i in progressinfo(range(len(test_beats)/10)):
            try:
                Z = Z + list(predict(test_beats[range(i*10,(i+1)*10)]))
                #print str(i) + " of " + str(len(test_beats[0])/10)
            except:
                break
                last = (i+1)*10
            last = (i+1)*10
            
        
        from numpy import min
        aux = test_beats[min([last,len(test_beats)]):]
        if len(aux) > 0:
            Z = Z + list(predict(array(aux)))
        return Z


class clf_LDA(__basic__):
    
    def __init__(self,n_components, train_data, train_labels):
        from scikits.learn.lda import LDA
        self.lda = LDA(n_components)
        self.lda.fit(train_data, train_labels)
        
    def predict(self, test_data):
        return self._clf_executor(self.lda.predict,test_data)

class clf_SVM(__basic__):

    def __init__(self,train_data, train_labels):
        from sklearn.svm import SVC
        #import ipdb
        self.clf = SVC(cache_size = 1500)
        self.clf.fit(train_data, train_labels)

    def predict(self, test_data):        
        return self._clf_executor(self.clf.predict,test_data)

class clf_NearestNeighbors(__basic__):

    def __init__(self,n_neighbors, train_data, train_labels):
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = KNeighborsClassifier(n_neighbors)
        self.clf.fit(train_data, train_labels)
        
    def predict(self, test_data):
        return self._clf_executor(self.clf.predict,test_data)

'''
class clf_NN(__basic__):

    def __init__(self,train_data, train_labels):
   
        print "Entrenando con " + str(len(train_data)) + " casos"
        
        from numpy import abs, max,array,min
        
        alpha = max(array(zip(max(abs(train_data),1), max(abs(test_data[lead]),1))),1)
        train_data_norm = [[x/alpha[i] for i,x in enumerate(y)] for y in train_data]
        test_data_norm = [[x/alpha[i] for i,x in enumerate(y)] for y in test_data[lead]]
        
   
        from pybrain.datasets import SupervisedDataSet
        from pybrain.structure import LinearLayer, SigmoidLayer, GaussianLayer
        from pybrain.structure import FeedForwardNetwork
        from pybrain.structure import FullConnection
        
        from pybrain.datasets import ClassificationDataSet
        from pybrain.datasets.unsupervised import UnsupervisedDataSet
        
        cs = UnsupervisedDataSet(len(test_data[lead][0]))
        for i,x in enumerate(test_data_norm):
            cs.addSample(test_data_norm[i])
        
        ds = SupervisedDataSet(len(test_data[lead][0]),1)
        for i,x in enumerate(train_data_norm):
            ds.addSample(tuple(train_data_norm[i]), (train_labels[i],))
        
        n = FeedForwardNetwork()
        
        inLayer = LinearLayer(len(test_data[lead][0]))
        hiddenLayer = GaussianLayer(6)
        #hiddenLayer = LinearLayer(1)
        outLayer = LinearLayer(1)
        
        n.addInputModule(inLayer)
        n.addModule(hiddenLayer)
        n.addOutputModule(outLayer)
        
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        
        n.addConnection(in_to_hidden)
        n.addConnection(hidden_to_out)
            
        n.sortModules()
        
        
        from pybrain.supervised.trainers import BackpropTrainer
        trainer = BackpropTrainer(n, ds,momentum=0.1, verbose=True, weightdecay=0.01)
        trainer.trainUntilConvergence(maxEpochs = 50, continueEpochs = 50, validationProportion = 0.05)
        
        out = n.activateOnDataset(cs)
        out = out.argmax(axis=1)  # the highest output activation gives the class
    
        return out
    
'''    
    
    
    

