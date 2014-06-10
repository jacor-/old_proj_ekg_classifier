


class thrs:
    def __init__(self, input_wave):
        from numpy import mod, array, sqrt, dot,median,convolve,max
        self.D0 = 20
        self.D1 = 25
        self.D2 = 70
        self.last_det = 0
        self.mu = 0.45
        self.a_up = 0.2
        self.a_down = 0.6
        self.z_cumulative = 0.5
        self.n_max = max(input_wave[:1000])
        self.input_wave = input_wave
        
    def setNewPos(self, pos):
        self.last_det = pos
        from numpy import max
        self.n_max = max(self.input_wave[max([pos-400,0]):min([pos+1000,len(self.input_wave)])])*1.1
        if self.input_wave[pos]-self.z_cumulative > 0:
            self.z_cumulative = float(self.z_cumulative + self.a_up * (self.input_wave[pos]-self.z_cumulative))
        else:
            self.z_cumulative = float(self.z_cumulative + self.a_down * (self.input_wave[pos]-self.z_cumulative))
        return 
    
    def getThrs(self, pos):
        if pos-self.last_det < self.D0:
            return self.n_max
        elif pos-self.last_det < self.D1:
            return  self.n_max - (self.n_max-self.z_cumulative)/(self.D1-self.D0)*(pos-self.last_det-self.D0)
        elif pos-self.last_det < self.D2:
            return self.mu * self.z_cumulative
        else:
            return self.mu*self.D2/(pos-self.last_det) * self.z_cumulative
        
        
            
        
            
    
    
    
    
    
    
    
    