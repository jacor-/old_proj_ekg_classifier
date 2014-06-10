import beatWork.characterization.production as chrClass
import cPickle

class trainer:
    
    def Load(self):
        f = open(self.nom,'rb')
        tmp_dict = cPickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 

    def __init__(self, nom):
        self.nom = nom
        self.clf_complete = False
        self.chr_complete = False
        self.train_data = []
        self.train_labels = []

    def loadClassifier(self,clf_name):
        import cPickle
        f = open(clf_name,'rb')
        self.clf = cPickle.load(f)
        self.clf_complete = True
    
    def setClassifier(self,clf_class):
        self.clf = clf_class
        self.clf_complete = True
        
    def trainClassifier(self,clf_class, train_data, train_labels):
        if self.chr_complete == False:
            print "You must define a characterization system"
            return
        self.clf = clf_class(train_data,train_labels)
        self.clf_complete = True
    
    def loadCharacterizer(self, characterizer_name):
        import cPickle
        f = open(characterizer_name,'rb')
        self.chr = cPickle.load(f)
        self.chr_complete = True     
        
    def setCharacterizer(self, characterizer):
        import cPickle
        self.chr = characterizer
        self.chr_complete = True
    
    def saveInstance(self):
        import cPickle
        f = open(self.nom,'wb')
        cPickle.dump(self.__dict__,f)
        f.close()
    
    def runCharAndPrediction(self, sig,valid_labs):
        if self.chr_complete == False or self.clf_complete == False:
            if self.chr_complete == False:
                print "You must define a characterization system before run a test"
            if self.clf_complete == False:
                print "You must define a classificator system before run a test"
            return " ERROR "
        ttest2 = self.chr.characterize(sig, valid_labs)
        return self.clf.predict(ttest2)

    def runPrediction(self, ttest2):
        if self.chr_complete == False or self.clf_complete == False:
            if self.clf_complete == False:
                print "You must define a classificator system before run a test"
            return " ERROR "
        return self.clf.predict(ttest2)
        


