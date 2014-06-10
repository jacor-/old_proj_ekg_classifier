
class FischerScore:
    @staticmethod
    def require_labels():
        return True

    def __init__(self,n_components, train_data, train_label):
        from my_methods import fischerScore
        from numpy import array
        reload(fischerScore)
        self.dr = fischerScore.fischerScore(n_components)
        self.dr.fit(train_data,  train_label, True)
    
    def reduce(self,data):
        return self.dr.transform(data)

class PCA:
    @staticmethod
    def require_labels():
        return False
    
    def __init__(self,n_components, train_data):
        import mdp 
        self.pca = mdp.nodes.PCANode(output_dim=n_components)
        self.pca.train(train_data)
        self.pca.stop_training()
        
    def reduce(self,data):
        return self.pca.execute(data)

class LDA:
    @staticmethod
    def require_labels():
        return True
    
    def __init__(self,n_components, train_data, train_label):
        from sklearn.lda import LDA
        self.lda = LDA(n_components)
        self.lda.fit(train_data, train_label)

    def reduce(self,data):
        return self.lda.transform(data)
        
class QDA:
    @staticmethod
    def require_labels():
        return True
    
    def __init__(self, n_components, train_data, train_label):
        from sklearn.qda import QDA
        self.lda = QDA(n_components)
        self.lda.fit(train_data, train_label)
        
    def reduce(self,data):
        return self.lda.transform(data)
        
class ICA:
    @staticmethod
    def require_labels():
        return False

    def __init__(self,n_param, train_data):
        import mdp 
        self.ica = mdp.nodes.FastICANode()
        self.ica.train(train_data)
        self.ica.stop_training()
        
    def reduce(self,data):
        return self.ica.execute(data)
        
    
