
class fischerScore():
    '''
    Just train and execute a fischerScore test. At the end of the test you will have 3 values:
    - weight: value of each new base
    - p_possibles: vectors of the new base (in fact, which components are more discriminant
    - transform: the best of the new dimension (vector in p_possibles with the higher weight)
    '''
    
    
    def _fact(self,A, limit):
        if A == limit:
            return 1
        if A == 1:
            return 1
        return A*self._fact(A-1,limit)
    def _fact_iter(self,A,limit):
        aux = 1
        while 1:
            if A == 0:
                break
            if A == limit:
                break
            aux = aux * A
            A = A - 1
        return aux
    
    def _combinatoria(self,A,B):
        return float(self._fact_iter(B,max(B-A,A)))/self._fact_iter(min(B-A,A),1)
    
    
    def __init__(self, n_components):
        self.n_components = n_components
    
            
    
    def fit(self, data, labels, fake = True):
        from numpy import zeros,arange,divide,sum,mean,argmax
        list_labels = list(set(labels))
        list_labels = dict(zip(list_labels, arange(len(list_labels))))
        mean_per_label = zeros([len(list_labels),len(data[0])])
        quant_per_label = zeros(len(list_labels))
        for index_data, label in enumerate(labels):
            quant_per_label[list_labels[label]] = quant_per_label[list_labels[label]] + 1
            mean_per_label[list_labels[label]] = mean_per_label[list_labels[label]]+data[index_data]
    
        #general_mean = sum(mean_per_label,0)
        general_mean = mean(data,0)
        
        for i,x in enumerate(quant_per_label):
            mean_per_label[i] = mean_per_label[i]/x

        St = data-general_mean
        St = sum(St * St,0)
        
        Sb = (mean_per_label-general_mean)
        for i,x in enumerate(quant_per_label):
            Sb[i] = Sb[i]*x
        
        Sb = sum(Sb*Sb,0)
        
        
        
        
        if not fake:
            from itertools import combinations
            poss = combinations(range(len(data[0])), self.n_components)
            from numpy import sum
            possibilities = sum([1 for index_1 in poss])
            poss = combinations(range(len(data[0])), self.n_components)
            self.p_possibles = zeros([possibilities,len(data[0])])
            for index_1 in range(len(self.p_possibles)):
                aux = poss.next()
                for index_2 in aux:
                    try:
                        self.p_possibles[index_1][index_2] = 1
                    except:
                        pass
            
            from numpy import dot,isnan
            St_w = dot(self.p_possibles,St)
            Sb_w = dot(self.p_possibles,Sb)
            self.weights = divide(Sb_w,St_w)
            for i,x in enumerate(St_w):
                if x == 0:
                    self.weights[i] = 0
            self.__transform = self.p_possibles[argmax(self.weights)]
        else:
            from itertools import combinations
            poss = combinations(range(len(data[0])), 1)
            from numpy import sum
            possibilities = sum([1 for index_1 in poss])
            poss = combinations(range(len(data[0])), 1)
            self.p_possibles = zeros([possibilities,len(data[0])])
            for index_1 in range(len(self.p_possibles)):
                aux = poss.next()
                for index_2 in aux:
                    try:
                        self.p_possibles[index_1][index_2] = 1
                    except:
                        pass
            
            from numpy import dot,isnan
            St_w = dot(self.p_possibles,St)
            Sb_w = dot(self.p_possibles,Sb)
            self.weights = divide(Sb_w,St_w)
            
            for i,x in enumerate(St_w):
                if x == 0:
                    self.weights[i] = 0
                    
            self.__transform = zeros(len(self.p_possibles[0]))
            cosa = sorted(zip(self.weights, range(len(self.weights))),reverse = True, key = lambda x:x[0])
            for x in range(self.n_components):
                self.__transform = self.__transform + self.p_possibles[cosa[x][1]]
            
                
    def transform(self,data):
        from numpy import take
        ax = [i for i in range(len(self.__transform)) if self.__transform[i] == 1]
        return take(data,ax,axis =1)
    
    def transformFake(self,data,quant = 2):
        from numpy import take
        ax = [i for i in range(len(self.__transform)) if self.__transform[i] == 1]
        return take(data,ax,axis =1)

    
if __name__ == "__main__":
    from numpy import array
    data = array([[0, 1,1,1,  2],
                  [-1,1,1,1,  4],
                  [1, 1,1,1,  4.1],
                  [10,1,1,1,  4.2],
                  [0, 1,1,0.9,1.9],
                  [0, 1,1,0.8,2.1]])
    
    labels = [0,1,1,1,0,0]
    
    fs = fischerScore(4)
    fs.train(data, labels)
    sol = fs.execute(data)
    print str(sol)

    
        
    