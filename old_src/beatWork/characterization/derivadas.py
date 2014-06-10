 
def characterizeCosaCovAll(Xs_C1,args):
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean

    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean
    from numpy import cov,mean,array

    if len(args[1]) == 0:
        Ys = args[0]
        
        normals = [x for i,x in enumerate(Xs_C1) if Ys[i] == 0]
        Ns = 200000
        Cl = 100
    
        from numpy import zeros, log10,array,mean,array
        from scipy.cluster.vq import kmeans, kmeans2
        from numpy import zeros
        
        a,b = kmeans2(array(normals), Cl)
        quant = [int(round(float(sum(b==i))/len(normals)*Ns)) for i in range(Cl)]
        quant2 = [max([log10(x),0]) for x in quant]
        quant3n = [int(round(float(x)/sum(quant2)*Ns))  for x in quant2]
        sels = zeros(Cl)
        subsels_norm = []
        for i,x in enumerate(b):
            if sels[x] <= quant3n[x]:
                subsels_norm.append(normals[i])
                sels[x] = sels[x] + 1
            if len(subsels_norm) >= Ns:
                break
        
    
        ventriculars = [x for i,x in enumerate(Xs_C1) if Ys[i] == 1]
    
        
        
        me = mean(array(ventriculars+subsels_norm),0)
        covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    else:
 
        me = args[1][0]
        covar = args[1][1]
    #covar = inv(cov(array(subsels_norm).transpose()))
    return [mahalanobis(x,me,covar) for x in Xs_C1], me, covar


def characterizeCosaCovNormal(Xs_C1,args):
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean

    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean
    from numpy import cov,mean,array

    if len(args[1]) == 0:
        Ys = args[0]
        normals = [x for i,x in enumerate(Xs_C1) if Ys[i] == 0]
        Ns = 200000
        Cl = 100
    
        from numpy import zeros, log10,array,mean,array
        from scipy.cluster.vq import kmeans, kmeans2
        from numpy import zeros
    
        a,b = kmeans2(array(normals), Cl)
        quant = [int(round(float(sum(b==i))/len(normals)*Ns)) for i in range(Cl)]
        quant2 = [max([log10(x),0]) for x in quant]
        quant3n = [int(round(float(x)/sum(quant2)*Ns))  for x in quant2]
        sels = zeros(Cl)
        subsels_norm = []
        for i,x in enumerate(b):
            if sels[x] <= quant3n[x]:
                subsels_norm.append(normals[i])
                sels[x] = sels[x] + 1
            if len(subsels_norm) >= Ns:
                break
        me = mean(array(subsels_norm),0)
        covar = inv(cov(array(subsels_norm).transpose()))
    else:
        me = args[1][0]
        covar = args[1][1]

    #covar = inv(cov(array(subsels_norm).transpose()))
    return [mahalanobis(x,me,covar) for x in Xs_C1], me, covar

