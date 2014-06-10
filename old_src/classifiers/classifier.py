
class Classifier:
    def __init__(self,clasifier, nom = "last_classifier"):
        self.clasifier = clasifier
        self.nom = nom
            
    def set_train(self,clf_file_name):
        import cPickle
        f = open(clf_file_name, 'rb')
        self.clf = cPickle.load(f)
        f.close()
                
    def predict(self, data):
        return self.clf.predict(data)
    
    def getMatrixResult(self,indexs,Z_ref,Z_res):
        res = []
        for i in range(len(indexs)-1):
            Z_ref_loc = Z_ref[indexs[i]:indexs[i+1]]
            Z_res_loc = Z_res[indexs[i]:indexs[i+1]]
            FP = sum(Z_res_loc-Z_ref_loc==1)
            FN = sum(Z_res_loc-Z_ref_loc==-1)
            res.append([len(Z_ref_loc),sum(Z_ref_loc),len(Z_ref_loc)-sum(Z_ref_loc),sum(Z_ref_loc)-FN,len(Z_ref_loc)-sum(Z_ref_loc)-FP,FP,FN])
        return res


    def fes_i_guarda_la_prediccio(self,Xs_C1,lead,case,indexs):
        from numpy import array
        import cPickle
        f=open('resultats/resultatsClassificacio/'+self.nom+'/'+case+'_'+str(lead),'wb')
        Z = self.predict(Xs_C1)
        cPickle.dump(Z,f)
        f.close()
        return Z

    def guardaResultats(self,lead,case,Ys):
        from numpy import array
        import cPickle        
        
        f=open('resultats/resultatsClassificacio/'+self.nom+'/'+case+'_'+str(lead),'rb')
        Z = cPickle.load(f)
        f.close()
        
        f=open('resultats/resultatsClassificacio/'+self.nom+'/resultats_macos_'+case+'_'+str(lead),'rb')
        cPickle.dump(getMatrixResult([0,len(Ys)],Ys,Z),f)
        f.close()
    
   
   
    