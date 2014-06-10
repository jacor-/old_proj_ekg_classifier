

class thrs:
    def __init__(self, input_wave):
        from numpy import mod, array, sqrt, dot,median,convolve
        self.D0 = 20
        self.last_det = 0
        self.mu = 0.6
        self.a_up = 0.2
        self.a_down = 0.6
        self.z_cumulative = 10
        self.n_max = max(input_wave[:1000])
        self.input_wave = input_wave
        self.rr = 60
        self.setNewPos(0)
        self.lmbda = 10
        
    def setNewPos(self, pos):
        self.rr = pos-self.last_det
        self.last_det = pos
        from numpy import max
        self.n_max = max(self.input_wave[max([pos-400,0]):min([pos+1000,len(self.input_wave)])])*1.1
        if self.input_wave[pos]-self.z_cumulative > 0:
            self.z_cumulative = float(self.z_cumulative + self.a_up * (self.input_wave[pos]-self.z_cumulative))
        else:
            self.z_cumulative = float(self.z_cumulative + self.a_down * (self.input_wave[pos]-self.z_cumulative))

        from numpy import log,e
        
        lmbda2 = log(self.mu)/((self.D0-self.rr)/2)
        from numpy import isinf
        if not isinf(lmbda2):
            self.lmbda = lmbda2
            self.A = self.z_cumulative/e**(-self.lmbda*self.D0)
        return 
    
    def getThrs(self, pos):
        if pos-self.last_det < self.D0:
            return self.n_max
        #elif pos-self.last_det < self.D1:
        #    return  self.n_max - (self.n_max-self.z_cumulative)/(self.D1-self.D1)(pos-self.last_det)
        else:
            from numpy import e
            return self.A * e**(-self.lmbda * (pos-self.last_det))
        
        
            
        
            
    