


class dimensionalityReductor:
    def __init__(self, dimens_reductors, ttrain, tlabels):
        self.transformators = []
        from numpy import array
        anterior_data = array(ttrain)
        for reductor in dimens_reductors:
            if reductor['algorithm'].require_labels() == True:
                self.transformators.append(reductor['algorithm'](reductor['param'], anterior_data, tlabels))
            else:
                self.transformators.append(reductor['algorithm'](reductor['param'], anterior_data))
            anterior_data = self.transformators[-1].reduce(ttrain)
        
    def reduce(self, data):
        for transf in self.transformators:
            data = transf.reduce(data)
        return data
'''
[{'algorithm':PCA, 'param':14},{'algorithm':FischerScore, 'param':8}]
'''
