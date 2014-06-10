
class production_characterization_system:
    def __init__(self):
        self.pipeline = []
        self.arguments = []
        
    def addPipeline(self, f, args):
        self.pipeline.append(f)
        self.arguments.append(args)
        
    def characterize(self, sig, data):
        from numpy import array
        for i,x in enumerate(zip(self.pipeline,self.arguments)):
            if i == 0:
                Z = array(x[0](sig,data))
            else:
                #import ipdb
                #ipdb.set_trace()

                if len(x[1]) > 0:
                    Z2 = x[0](Z,x[1])
                else:
                    Z2 = x[0](Z)
                Z = list(Z.transpose())
                Z.append(Z2[0])
                Z = array(Z).transpose()
        return Z
        
