

class thrs:
    def __init__(self,input_wave):
        from numpy import mod, array, sqrt, dot,median,convolve,mean
        steps = len(input_wave)/1000
        
        w = array([1,2,1])
        w = w/sqrt(dot(w,w))
        
        mu = 0.6
        
        if steps * 1000 != len(input_wave):
            steps = steps + 1
        
        sigma = [mu*mean(input_wave[step*1000:(step+1)*1000])*2 for step in range(steps)]
        tau = convolve(sigma, w)[1:-1]
        tot = []
        for x in tau:
            tot = tot + [x]*1000
        self.thrs = tot
        
        self.D0 = 20
        self.last_pos = 0
    
    def setNewPos(self, pos):
        self.last_pos = pos 
        return 
    
    def getThrs(self, pos):
        if pos-self.last_pos < self.D0:
            return 10000.
        else:        
            return self.thrs[pos]
    
    

